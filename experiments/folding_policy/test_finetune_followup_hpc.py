"""Bounded follow-up scheduling and independently frozen source provenance."""
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

import finetune_followup_launch as launch
import finetune_launch
from geometry import contacts
from test_inference_hpc import make_source_fixture


def make_finetune_source(root):
    """Synthetic receipts/checkpoint bytes for protocol tests, never model inference."""
    root = Path(root)
    allocation, source = root / 'allocation', root / 'source'
    make_source_fixture(allocation)
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = finetune_launch.prepare(Namespace(directory=str(source), source_allocation=str(allocation),
                                                     python=sys.executable, dry_run=True))
    data = json.loads((source / 'data.json').read_text())
    for task in manifest['tasks']:
        arm, seed = task['arm'], task['seed']
        folder = finetune_launch.task_directory(source, task)
        folder.mkdir()
        metadata = {
            'schema': 1, 'kind': 'allocation_training_state' if arm == 'parent' else 'finetune_training_state',
            'arm': None if arm == 'parent' else arm, 'seed': seed, 'threads': 4,
            'dataset_sha256': manifest['hashes']['data.json'],
            'state_sha256': task['parent_state_sha256'] if arm == 'parent' else f'{arm}-{seed}-state',
            'training_seconds': task['lineage_training_seconds'],
            'mode': 'patternboost' if arm == 'parent' else 'finetune',
            'stage': 'patternboost' if arm == 'parent' else arm,
            'condition': 'patternboost' if arm == 'parent' else arm,
        }
        digest = task['parent_sha256']
        if arm != 'parent':
            training = folder / 'training'
            training.mkdir()
            checkpoint = training / 'final.pt'
            checkpoint.write_text(f'Synthetic protocol test checkpoint {arm} {seed}\n')
            digest = launch.sha(checkpoint)
            counters = {'supervised_updates': 2, 'rl_updates': 2, 'preference_updates': 2,
                        'generated_folds': 8, 'valid_folds': 7, 'rl_groups': 2, 'zero_advantage_groups': 0}
            record = {'filename': 'final.pt', 'file_sha256': digest, 'state_sha256': metadata['state_sha256'],
                      'training_seconds': 9000, 'selected': False}
            trained = {
                'status': 'complete', 'arm': arm, 'seed': seed, 'threads': 4, 'budget_s': 1800,
                'base_training_seconds': 7200, 'additional_training_seconds': 1800,
                'training_seconds': 9000, 'dataset_sha256': manifest['hashes']['data.json'],
                'parent_file_sha256': task['parent_sha256'],
                'parent_state_sha256': task['parent_state_sha256'],
                'initial_state_sha256': task['parent_state_sha256'],
                'reference_state_sha256': task['parent_state_sha256'],
                'final_state_sha256': metadata['state_sha256'],
                'final_test_enabled': False, 'test_evaluated': False, 'selection': {'selected': False},
                'training_settings': {'learning_rate': task['learning_rate']},
                'compute_elapsed_s': 1800.1, 'segment_counters': counters, 'checkpoint_records': [record],
                'handoff': {'pretraining_sha256': task['parent_state_sha256'],
                            'finetune_initial_sha256': task['parent_state_sha256'],
                            'pretraining_optimizer_sha256': task['parent_optimizer_sha256'],
                            'learning_rate_override': task['learning_rate'],
                            'optimizer_moments_retained': True, 'rng_state_retained': True,
                            'elite_pool_retained': True},
            }
            launch.write_json(training / 'result.json', trained)
            metadata.update(base_training_seconds=7200, additional_training_seconds=1800,
                            parent_file_sha256=task['parent_sha256'],
                            parent_state_sha256=task['parent_state_sha256'], segment_counters=counters)
        rows = []
        for sequence in data['validation']:
            n = len(sequence['sequence'])
            radius = n // 2
            positions = ([[x, 0, 0] for x in range(radius + 1)]
                         + [[x, 1, 0] for x in range(radius, -1, -1)])[:n]
            score = contacts(sequence['sequence'], positions)
            rows.append({**sequence, 'length': n, 'method': 'model', 'seed': seed,
                         'budget_s': 60, 'elapsed_s': 60.1, 'setup_s': .1,
                         'generated_complete_folds': 1, 'attempted_folds': 1, 'valid_fold_rate': 1.0,
                         'best_contacts': score, 'best_positions': positions,
                         'curve': [{'seconds': 1, 'contacts': score, 'positions': positions}],
                         'checkpoint_stage': metadata['stage'], 'checkpoint_condition': metadata['condition'],
                         'checkpoint_seed': seed, 'checkpoint_dataset_sha256': manifest['hashes']['data.json'],
                         'proof_within_budget': False})
        launch.write_json(folder / 'launch.json', {'status': 'complete', 'task': task,
                                                  'software': manifest['software_at_freeze']})
        launch.write_json(folder / 'evaluation.json', {
            'schema': 1, 'kind': 'finetune_endpoint_evaluation', 'status': 'complete',
            'split': 'validation', 'final_test_access': False, 'selection': 'fixed_training_endpoint',
            'method': 'model', 'arm': arm, 'seed': seed, 'threads': 4,
            'ids': manifest['validation_ids'], 'dataset_file_sha256': manifest['hashes']['data.json'],
            'evaluation_seconds_per_sequence_per_seed': 60, 'model_setup_charged_per_sequence': True,
            'checkpoint_sha256': digest, 'checkpoint_metadata': metadata, 'results': rows,
        })
    return source


class FollowupHpcTests(unittest.TestCase):
    def fixture(self, root):
        source = make_finetune_source(root)
        before = {str(p.relative_to(source)): launch.sha(p) for p in source.rglob('*') if p.is_file()}
        run = root / 'followup'
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = launch.prepare(Namespace(directory=str(run), source_finetune=str(source),
                                                python=sys.executable, dry_run=True))
        self.assertEqual(before, {str(p.relative_to(source)): launch.sha(p)
                                  for p in source.rglob('*') if p.is_file()})
        return source, run, manifest

    def test_only_six_new_controls_all_twenty_four_endpoints_and_exact_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, manifest = self.fixture(Path(directory))
            rows = manifest['tasks']
            self.assertEqual(len(rows), 24)
            self.assertEqual([t['arm'] for t in rows], [a for a in launch.ARMS for _ in launch.SEEDS])
            self.assertEqual(sum(t['new_training_seconds'] * 4 / 3600 for t in rows), 12)
            self.assertEqual(sum(t['evaluation_seconds'] * 16 * 4 for t in rows) / 3600, 256)
            self.assertEqual(sum(t['slurm_seconds'] * 4 for t in rows) / 3600, 384)
            self.assertEqual(manifest['attempt_checkpoints'], [128, 512, 1024])
            self.assertTrue(all(t['new_training_seconds'] == 0 and t['checkpoint'] for t in rows[:18]))
            self.assertTrue(all(t['new_training_seconds'] == 1800 and t['checkpoint'] is None
                                and t['learning_rate'] == .00003 for t in rows[18:]))
            for seed in launch.SEEDS:
                self.assertEqual(len({t['parent_sha256'] for t in rows if t['seed'] == seed}), 1)
            shutil.rmtree(source)
            shutil.rmtree(Path(directory) / 'allocation')
            self.assertEqual(launch.verify(run), manifest)

    def test_frozen_receipt_endpoint_and_implementation_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, manifest = self.fixture(Path(directory))
            for relative in (manifest['tasks'][3]['checkpoint'], manifest['tasks'][3]['training_receipt'],
                             'bundle/folding_policy/model.py', 'source_finetune/finetune.json'):
                path = run / relative
                original = path.read_bytes()
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                    launch.verify(run)
                path.write_bytes(original)

    def test_resigning_does_not_permit_retraining_reused_endpoints_or_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, manifest = self.fixture(Path(directory))
            manifest['tasks'][3]['new_training_seconds'] = 1800
            launch.write_json(run / 'finetune_followup.json', manifest)
            (run / 'finetune_followup.sha256').write_text(launch.sha(run / 'finetune_followup.json'))
            with self.assertRaisesRegex(ValueError, 'task plan or endpoint provenance'):
                launch.verify(run)
            manifest['tasks'][3]['new_training_seconds'] = 0
            core = run / 'bundle/folding_policy/model.py'
            core.write_text(core.read_text() + '\n# changed\n')
            manifest['hashes']['bundle/folding_policy/model.py'] = launch.sha(core)
            launch.write_json(run / 'finetune_followup.json', manifest)
            (run / 'finetune_followup.sha256').write_text(launch.sha(run / 'finetune_followup.json'))
            with self.assertRaisesRegex(ValueError, 'Source model or training implementation changed'):
                launch.verify(run)

    def test_incomplete_or_wrong_software_source_task_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_finetune_source(Path(directory))
            path = source / 'tasks/000_patternboost_seed0/launch.json'
            original = json.loads(path.read_text())
            for changed in ({**original, 'status': 'failed'}, {**original, 'software': {}}):
                launch.write_json(path, changed)
                with self.assertRaisesRegex(ValueError, 'incomplete or its runtime provenance changed'):
                    launch.validate_source(source)

    def test_source_wrong_lr_dropped_rng_invalid_witness_and_missing_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_finetune_source(Path(directory))
            training = source / 'tasks/003_grpo_seed0/training/result.json'
            original = training.read_text()
            value = json.loads(original)
            value['training_settings']['learning_rate'] = .00003
            launch.write_json(training, value)
            with self.assertRaisesRegex(ValueError, 'settings'):
                launch.validate_source(source)
            training.write_text(original)
            value = json.loads(original)
            value['handoff']['rng_state_retained'] = False
            launch.write_json(training, value)
            with self.assertRaisesRegex(ValueError, 'handoff'):
                launch.validate_source(source)
            training.write_text(original)
            path = source / 'tasks/003_grpo_seed0/evaluation.json'
            original = path.read_text()
            value = json.loads(original)
            value['results'][0]['best_contacts'] += 1
            launch.write_json(path, value)
            with self.assertRaisesRegex(ValueError, 'witness'):
                launch.validate_source(source)
            value = json.loads(original)
            value['results'].pop()
            launch.write_json(path, value)
            with self.assertRaisesRegex(ValueError, 'every validation sequence'):
                launch.validate_source(source)

    def test_runtime_drift_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_finetune_source(root)
            with patch.object(launch, 'software', return_value={}):
                with self.assertRaisesRegex(ValueError, 'versions differ'):
                    launch.prepare(Namespace(directory=str(root / 'followup'), source_finetune=str(source),
                                             python=sys.executable, dry_run=True))
            self.assertFalse((root / 'followup').exists())

    def test_single_concurrent_array_resources_email_and_no_dependencies(self):
        with patch.dict(launch.os.environ, {'HP_MAIL_USER': 'user@example.org'}):
            command = launch.submission_command('/cluster/followup')
        for value in ('--array=0-23%24', '--time=240', '--mail-user=user@example.org'):
            self.assertIn(value, command)
        self.assertFalse(any(arg.startswith('--dependency') for arg in command))
        script = (launch.HERE / 'finetune_followup.sbatch').read_text()
        for value in ('--cpus-per-task=4', '--mem=16G', '--constraint=ehc', '--partition=compute',
                      '--mail-type=END,FAIL', '--no-requeue', '--cpu-bind=cores'):
            self.assertIn(value, script)

    def test_accepted_or_ambiguous_submission_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(launch.subprocess, 'run', return_value=Namespace(stdout='501;cluster\n', stderr='')) as submit:
                with contextlib.redirect_stdout(io.StringIO()):
                    record = launch.submit_array(run)
                self.assertEqual(record['job_id'], '501')
                self.assertEqual((run / 'job_ids.txt').read_text(), '501\n')
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    launch.submit_array(run)
                self.assertEqual(submit.call_count, 1)
        for response in (Namespace(stdout='uncertain', stderr=''),
                         subprocess.CalledProcessError(1, ['sbatch'], output='quota', stderr='busy')):
            with tempfile.TemporaryDirectory() as directory:
                kwargs = {'side_effect': response} if isinstance(response, Exception) else {'return_value': response}
                with patch.object(launch.subprocess, 'run', **kwargs) as submit:
                    with self.assertRaisesRegex(RuntimeError, 'check Slurm'):
                        launch.submit_array(Path(directory))
                    with self.assertRaisesRegex(ValueError, 'already attempted'):
                        launch.submit_array(Path(directory))
                    self.assertEqual(submit.call_count, 1)

    def test_dry_run_existing_and_nested_output_never_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launch.subprocess, 'run') as submit, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.submit_array(root, dry_run=True)['status'], 'prepared')
            submit.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                launch.prepare(Namespace(directory=directory, source_finetune='/unused',
                                         python=sys.executable, dry_run=True))
            with self.assertRaisesRegex(ValueError, 'outside the completed source'):
                launch.prepare(Namespace(directory=str(root / 'new'), source_finetune=directory,
                                         python=sys.executable, dry_run=True))


if __name__ == '__main__':
    unittest.main()
