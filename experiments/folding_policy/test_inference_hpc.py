"""Freeze and submission checks for the saved-policy validation comparison."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import allocation_launch
from dataset import make_dataset
import inference_launch as launch


def make_source_fixture(source):
    """A completed source with synthetic checkpoint bytes; never used for model inference."""
    source = Path(source)
    policy = source / 'bundle' / 'folding_policy'
    policy.mkdir(parents=True)
    for path in launch.HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.sbatch', '.md', '.txt'):
            shutil.copy2(path, policy / path.name)
    cluster = source / 'bundle' / 'cluster'
    cluster.mkdir()
    for name in ('hp_solver.py', 'geometry.py', 'protocol.json'):
        shutil.copy2(launch.HERE.parent / 'cluster' / name, cluster / name)
    data = make_dataset()
    launch.write_json(source / 'data.json', data)
    files = [path for path in (source / 'bundle').rglob('*') if path.is_file()] + [source / 'data.json']
    manifest = {'kind': 'validation_allocation_pilot', 'threads': 4, 'final_test_access': False,
                'no_automatic_followup': True, 'tasks': allocation_launch.tasks(),
                'software_at_freeze': launch.software(), 'source_revision': 'fixture-revision',
                'validation_ids': allocation_launch.validation_ids(data),
                'hashes': {str(path.relative_to(source)): launch.sha(path) for path in files}}
    launch.write_json(source / 'allocation.json', manifest)
    (source / 'allocation.sha256').write_text(launch.sha(source / 'allocation.json') + '\n')
    trained = {}
    for task in manifest['tasks']:
        arm, seed = task['arm'], task['seed']
        folder = source / 'tasks' / f'{arm}_seed{seed}'
        training = folder / 'training'
        training.mkdir(parents=True)
        launch.write_json(folder / 'launch.json', {'status': 'complete', 'task': task,
                                                 'software': manifest['software_at_freeze']})
        records = []
        for stage in task['evaluation_stages']:
            checkpoint = training / f'{stage}.pt'
            checkpoint.write_text(f'Synthetic test checkpoint: {arm}, {seed}, {stage}\n')
            records.append({'filename': checkpoint.name, 'file_sha256': launch.sha(checkpoint),
                            'state_sha256': f'state-{arm}-{seed}-{stage}',
                            'optimizer_sha256': f'optimizer-{arm}-{seed}-{stage}',
                            'training_seconds': task['stage_lineage_seconds'][stage], 'selected': False})
        receipt = {'status': 'complete', 'mode': arm if not task['parent_arm'] else 'rl_branch',
                   'seed': seed, 'threads': 4, 'budget_s': task['training_seconds'],
                   'training_seconds': 7200, 'compute_elapsed_s': task['training_seconds'] + .1,
                   'test_evaluated': False, 'final_test_enabled': False,
                   'selection': {'selected': False}, 'dataset_sha256': manifest['hashes']['data.json'],
                   'counters': {'supervised_updates': 1, 'rl_updates': 1},
                   'segment_counters': {'rl_updates': 1}, 'checkpoint_records': records}
        if task['parent_arm']:
            parent = next(row for row in trained[('patternboost', seed)]['checkpoint_records']
                          if row['filename'] == task['resume_stage'] + '.pt')
            state, optimizer = parent['state_sha256'], parent['optimizer_sha256']
            receipt.update(initial_state_sha256=state,
                           lineage={'parent_file_sha256': parent['file_sha256'],
                                    'parent_state_sha256': state, 'parent_optimizer_sha256': optimizer},
                           handoff={'pretraining_sha256': state, 'rl_initial_sha256': state,
                                    'pretraining_optimizer_sha256': optimizer, 'rl_initial_optimizer_sha256': optimizer,
                                    'optimizer_state_retained': True, 'rng_state_retained': True, 'elite_pool_retained': True})
        trained[(arm, seed)] = receipt
        launch.write_json(training / 'result.json', receipt)
    return manifest


class InferenceHpcTests(unittest.TestCase):
    def prepare_fixture(self, root):
        source, run = root / 'source', root / 'inference'
        make_source_fixture(source)
        before = {str(path.relative_to(source)): launch.sha(path) for path in source.rglob('*') if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = launch.prepare(Namespace(directory=str(run), source_allocation=str(source),
                                                python=sys.executable, dry_run=True))
        after = {str(path.relative_to(source)): launch.sha(path) for path in source.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        return source, run, manifest

    def test_frozen_budget_reuses_twelve_policies_and_six_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, manifest = self.prepare_fixture(Path(directory))
            self.assertEqual(len(manifest['tasks']), 18)
            self.assertEqual(len(manifest['validation_ids']), 16)
            rows = manifest['tasks']
            self.assertEqual(sum(row['seconds'] * 16 * 4 for row in rows) / 3600, 240)
            self.assertEqual(sum(row['slurm_seconds'] * 4 for row in rows) / 3600, 264)
            self.assertTrue(all(row['seconds'] == 600 and row['checkpoint'] for row in rows[:12]))
            self.assertTrue(all(row['seconds'] == 1050 and row['checkpoint'] is None for row in rows[12:]))
            self.assertEqual(launch.verify(run), manifest)
            self.assertEqual(launch.sha(source / 'bundle/cluster/hp_solver.py'),
                             launch.sha(run / 'bundle/cluster/hp_solver.py'))
            self.assertEqual(launch.sha(source / 'bundle/folding_policy/model.py'),
                             launch.sha(run / 'bundle/folding_policy/model.py'))
            shutil.rmtree(source)
            self.assertEqual(launch.verify(run), manifest)  # Fully self-contained after preparation.

    def test_checkpoint_and_receipt_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, manifest = self.prepare_fixture(Path(directory))
            checkpoint = run / manifest['tasks'][0]['checkpoint']
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                launch.verify(run)
            checkpoint.write_bytes(original)
            receipt = run / 'source_receipts/tasks/balanced_seed0/training/result.json'
            receipt.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                launch.verify(run)

    def test_source_rejects_failed_training_and_different_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            make_source_fixture(source)
            path = source / 'tasks/balanced_seed0/training/result.json'
            receipt = json.loads(path.read_text())
            receipt['status'] = 'failed'
            launch.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'did not complete correctly'):
                launch.validate_source(source)
            receipt['status'] = 'complete'
            receipt['handoff']['rl_initial_sha256'] = 'different weights'
            launch.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'shared parent weights'):
                launch.validate_source(source)

    def test_source_rejects_dropped_rng_and_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            make_source_fixture(source)
            path = source / 'tasks/more_pretrain_seed1/training/result.json'
            receipt = json.loads(path.read_text())
            receipt['handoff']['rng_state_retained'] = False
            launch.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'optimizer, RNG and elite pool'):
                launch.validate_source(source)
            receipt['handoff']['rng_state_retained'] = True
            launch.write_json(path, receipt)
            (source / 'tasks/patternboost_seed0/training/stage_7200.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'fixed training receipt'):
                launch.validate_source(source)

    def test_runtime_environment_must_match_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            make_source_fixture(source)
            with patch.object(launch, 'software', return_value={'python': 'other', 'packages': {}}):
                with self.assertRaisesRegex(ValueError, 'versions differ'):
                    launch.prepare(Namespace(directory=str(Path(directory) / 'new'), source_allocation=str(source),
                                             python=sys.executable, dry_run=True))
            self.assertFalse((Path(directory) / 'new').exists())

    def test_independent_arrays_resources_and_email(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.dict(launch.os.environ, {'HP_MAIL_USER': 'user@example.org'}):
                commands = [launch.submission_command(run, group) for group in launch.ARRAYS]
            self.assertIn('--array=0-11%12', commands[0])
            self.assertIn('--time=180', commands[0])
            self.assertIn('--array=12-17%6', commands[1])
            self.assertIn('--time=300', commands[1])
            self.assertTrue(all('--mail-user=user@example.org' in cmd for cmd in commands))
            self.assertFalse(any(arg.startswith('--dependency') for cmd in commands for arg in cmd))
            sbatch = (launch.HERE / 'inference.sbatch').read_text()
            for option in ('--cpus-per-task=4', '--mem=16G', '--constraint=ehc', '--mail-type=END,FAIL', '--no-requeue'):
                self.assertIn(option, sbatch)

    def test_partial_submission_preserves_first_array(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            failure = subprocess.CalledProcessError(1, ['sbatch'], output='quota', stderr='busy')
            with patch.object(launch.subprocess, 'run', side_effect=[Namespace(stdout='501;cluster\n', stderr='notice'), failure]):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'accepted arrays remain scheduled: 501'):
                    launch.submit_arrays(run)
            record = json.loads((run / 'submission.json').read_text())
            self.assertEqual(record['status'], 'partial_submission')
            self.assertEqual(record['arrays'][0]['sbatch_stderr'], 'notice')
            self.assertEqual(record['arrays'][1]['sbatch_stderr'], 'busy')
            self.assertEqual((run / 'job_ids.txt').read_text(), '501\n')

    def test_dry_run_never_submits_and_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(launch.subprocess, 'run') as submit, contextlib.redirect_stdout(io.StringIO()):
                record = launch.submit_arrays(Path(directory), dry_run=True)
            submit.assert_not_called()
            self.assertEqual(record['status'], 'prepared')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                launch.prepare(Namespace(directory=directory, source_allocation='/unused', python=sys.executable, dry_run=True))

    def test_output_cannot_mutate_source_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, 'outside the completed source'):
                launch.prepare(Namespace(directory=str(source / 'new'), source_allocation=directory,
                                         python=sys.executable, dry_run=True))


if __name__ == '__main__':
    unittest.main()
