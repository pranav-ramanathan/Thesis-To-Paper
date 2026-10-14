"""Freeze provenance, budgets and submission recovery for fine-tuning jobs."""
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

import finetune_launch as launch
from test_inference_hpc import make_source_fixture


class FinetuneHpcTests(unittest.TestCase):
    def prepare_fixture(self, root):
        source, run = root / 'source', root / 'finetune'
        make_source_fixture(source)
        before = {str(path.relative_to(source)): launch.sha(path)
                  for path in source.rglob('*') if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = launch.prepare(Namespace(directory=str(run), source_allocation=str(source),
                                                python=sys.executable, dry_run=True))
        after = {str(path.relative_to(source)): launch.sha(path)
                 for path in source.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        return source, run, manifest

    def test_freeze_has_five_branches_and_unchanged_parent_from_same_three_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, manifest = self.prepare_fixture(Path(directory))
            rows = manifest['tasks']
            self.assertEqual(len(rows), 18)
            self.assertEqual([row['arm'] for row in rows],
                             [arm for arm in (*launch.ARMS, 'parent') for _ in launch.SEEDS])
            self.assertEqual(sum(row['training_seconds'] * 4 for row in rows) / 3600, 30)
            self.assertEqual(sum(row['evaluation_seconds'] * 16 * 4 for row in rows) / 3600, 19.2)
            self.assertEqual(sum(row['slurm_seconds'] * 4 for row in rows) / 3600, 90)
            self.assertEqual(len(list((run / 'parents').glob('*.pt'))), 3)
            self.assertFalse(list((run / 'source_receipts').rglob('*.pt')))
            for seed in launch.SEEDS:
                matched = [row for row in rows if row['seed'] == seed]
                self.assertEqual(len({row['parent_sha256'] for row in matched}), 1)
                self.assertEqual(len({row['parent_state_sha256'] for row in matched}), 1)
                self.assertEqual(len({row['parent_optimizer_sha256'] for row in matched}), 1)
            self.assertTrue(all(row['training_seconds'] == 0 and row['lineage_training_seconds'] == 7200
                                for row in rows[15:]))
            self.assertTrue(all(row['training_seconds'] == 1800 and row['lineage_training_seconds'] == 9000
                                for row in rows[:15]))
            self.assertEqual(manifest, launch.verify(run))
            for relative in ('bundle/folding_policy/model.py', 'bundle/folding_policy/geometry.py',
                             'bundle/folding_policy/allocation_train.py', 'bundle/cluster/hp_solver.py'):
                self.assertEqual(launch.sha(source / relative), launch.sha(run / relative))
            shutil.rmtree(source)
            self.assertEqual(manifest, launch.verify(run))

    def test_checkpoint_receipt_and_frozen_code_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, manifest = self.prepare_fixture(Path(directory))
            paths = [run / manifest['tasks'][0]['parent'],
                     run / 'source_receipts/tasks/patternboost_seed0/training/result.json',
                     run / 'bundle/folding_policy/model.py']
            for path in paths:
                saved = path.read_bytes()
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                    launch.verify(run)
                path.write_bytes(saved)

    def test_resigning_changed_task_or_core_file_does_not_change_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, manifest = self.prepare_fixture(Path(directory))
            manifest['tasks'][0]['parent'] = manifest['tasks'][1]['parent']
            launch.write_json(run / 'finetune.json', manifest)
            (run / 'finetune.sha256').write_text(launch.sha(run / 'finetune.json'))
            with self.assertRaisesRegex(ValueError, 'task plan or parent provenance'):
                launch.verify(run)
            manifest['tasks'][0]['parent'] = 'parents/patternboost_seed0.pt'
            model = run / 'bundle/folding_policy/model.py'
            model.write_text(model.read_text() + '\n# changed\n')
            manifest['hashes']['bundle/folding_policy/model.py'] = launch.sha(model)
            launch.write_json(run / 'finetune.json', manifest)
            (run / 'finetune.sha256').write_text(launch.sha(run / 'finetune.json'))
            with self.assertRaisesRegex(ValueError, 'Source model or environment code changed'):
                launch.verify(run)

    def test_failed_or_selected_source_parent_cannot_start_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            make_source_fixture(source)
            path = source / 'tasks/patternboost_seed0/training/result.json'
            receipt = json.loads(path.read_text())
            receipt['status'] = 'failed'
            launch.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'did not complete correctly'):
                launch._source_parents(source)
            receipt['status'] = 'complete'
            receipt['selection']['selected'] = True
            launch.write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'budget, settings or software changed'):
                launch._source_parents(source)

    def test_runtime_versions_must_match_before_output_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            make_source_fixture(source)
            with patch.object(launch, 'software', return_value={'python': 'changed', 'packages': {}}):
                with self.assertRaisesRegex(ValueError, 'versions differ'):
                    launch.prepare(Namespace(directory=str(root / 'new'), source_allocation=str(source),
                                             python=sys.executable, dry_run=True))
            self.assertFalse((root / 'new').exists())

    def test_single_array_has_resources_email_and_no_dependency(self):
        with patch.dict(launch.os.environ, {'HP_MAIL_USER': 'user@example.org'}):
            command = launch.submission_command(Path('/cluster/frozen'))
        for arg in ('--array=0-17%18', '--time=75', '--mail-user=user@example.org'):
            self.assertIn(arg, command)
        self.assertFalse(any(arg.startswith('--dependency') for arg in command))
        script = (launch.HERE / 'finetune.sbatch').read_text()
        for value in ('--cpus-per-task=4', '--mem=16G', '--partition=compute',
                      '--constraint=ehc', '--mail-type=END,FAIL', '--no-requeue', '--cpu-bind=cores'):
            self.assertIn(value, script)

    def test_submission_records_job_and_blocks_second_submission(self):
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

    def test_ambiguous_response_and_sbatch_failure_are_saved_without_retry(self):
        for response in (Namespace(stdout='possibly accepted but no id', stderr='notice'),
                         subprocess.CalledProcessError(1, ['sbatch'], output='quota', stderr='busy')):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                kwargs = {'side_effect': response} if isinstance(response, Exception) else {'return_value': response}
                with patch.object(launch.subprocess, 'run', **kwargs) as submit:
                    with self.assertRaisesRegex(RuntimeError, 'check Slurm'):
                        launch.submit_array(run)
                    self.assertEqual(submit.call_count, 1)
                record = json.loads((run / 'submission.json').read_text())
                self.assertIn(record['status'], ('submission_uncertain', 'failed'))
                self.assertFalse(record['submitted'])
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    launch.submit_array(run)

    def test_dry_run_and_existing_or_nested_directory_never_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(launch.subprocess, 'run') as submit, contextlib.redirect_stdout(io.StringIO()):
                record = launch.submit_array(root, dry_run=True)
            submit.assert_not_called()
            self.assertEqual(record['status'], 'prepared')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                launch.prepare(Namespace(directory=directory, source_allocation='/unused',
                                         python=sys.executable, dry_run=True))
            with self.assertRaisesRegex(ValueError, 'outside the completed source'):
                launch.prepare(Namespace(directory=str(root / 'new'), source_allocation=directory,
                                         python=sys.executable, dry_run=True))


if __name__ == '__main__':
    unittest.main()
