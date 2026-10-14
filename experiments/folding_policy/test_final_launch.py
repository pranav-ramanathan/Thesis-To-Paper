"""Final campaign contracts; no final-set folds are generated in these tests."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import final_launch as launch
import final_worker as worker


class FinalLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hp-final-orchestration-')
        self.root = Path(self.temp.name)
        self.run = self.root / 'campaign'

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            launch.prepare(SimpleNamespace(directory=str(self.run), python=sys.executable, dry_run=True))
        return launch.verify(self.run)

    def test_frozen_matrix_dependencies_and_costs(self):
        manifest = self.prepare()
        self.assertEqual(len(manifest['tasks']), 55)
        self.assertEqual(len(manifest['evaluation_tasks']), 100)
        self.assertEqual(sum(t['training_seconds'] * 4 / 3600 for t in manifest['tasks']), 240)
        self.assertAlmostEqual(sum(len(t['sequence_ids']) * t['seconds'] * 4 / 3600
                                   for t in manifest['evaluation_tasks']), 8800 / 3)
        self.assertEqual(sum(t['slurm_minutes'] * 4 / 60 for t in manifest['tasks']), 3380)
        self.assertEqual(manifest['scientific_training_cpu_hours'], 400)
        data = json.loads((self.run / 'data.json').read_text())
        self.assertEqual([len(data[s]) for s in ('train', 'validation', 'test', 'benchmark')], [160, 16, 80, 8])
        commands = dict(launch.submission_commands(self.run, '12345'))
        self.assertEqual(len(commands), 11)
        for arm, command in commands.items():
            self.assertIn('--array=0-4%5', command)
            if arm in launch.BRANCHES:
                self.assertIn('--dependency=aftercorr:12345', command)
                self.assertIn('--kill-on-invalid-dep=yes', command)
            else:
                self.assertFalse(any(s.startswith('--dependency') for s in command))
            for seed in launch.SEEDS:
                task = launch.find_job(manifest, arm, seed)
                self.assertEqual(task['seed'], seed)
                if arm in launch.BRANCHES:
                    self.assertEqual(task['parent_task'], seed)
        self.assertIn(['grpo', 'scratch_grpo'], manifest['contrasts'])
        self.assertIn(['dpo', 'cp_sat'], manifest['contrasts'])
        self.assertEqual(json.loads((self.run / 'submission.json').read_text())['arrays'], [])

    def test_refuses_changed_source_manifest_and_destination(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.prepare()
        source = self.run / 'bundle/folding_policy/final_worker.py'
        original = source.read_text()
        source.write_text(original + '\n# altered\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            launch.verify(self.run)
        source.write_text(original)
        path = self.run / 'campaign.json'
        manifest = json.loads(path.read_text())
        manifest['evaluation_tasks'][0]['seconds'] = 24 * 3600
        launch.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            launch.verify(self.run)
        (self.run / 'campaign.sha256').write_text(launch.sha(path))
        with self.assertRaisesRegex(ValueError, 'matrix changed'):
            launch.verify(self.run)

    def test_partial_submission_preserves_accepted_ids_and_refuses_retry(self):
        self.run.mkdir()
        def submit(command, **kwargs):
            if '--job-name=hp-final-parent' in command:
                return SimpleNamespace(stdout='12345;apocrita\n', stderr='warning retained')
            raise subprocess.CalledProcessError(1, command, stderr='scheduler unavailable')
        with patch.object(launch, 'verify'), patch.object(launch.subprocess, 'run', side_effect=submit), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Partial or uncertain'):
                launch.submit(self.run)
            self.assertEqual((self.run / 'job_ids.txt').read_text().strip(), '12345')
            receipt = json.loads((self.run / 'submission.json').read_text())
            self.assertEqual(receipt['status'], 'submission_incomplete')
            self.assertEqual(receipt['arrays'][0]['job_id'], '12345')
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                launch.submit(self.run)

    def test_successful_submission_uses_actual_parent_and_retains_all_arrays(self):
        self.run.mkdir()
        calls = []
        def submit(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(stdout=str(800 + len(calls)) + '\n', stderr='')
        with patch.object(launch, 'verify'), patch.object(launch.subprocess, 'run', side_effect=submit), contextlib.redirect_stdout(io.StringIO()):
            result = launch.submit(self.run)
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual(len(calls), 11)
        for command in calls[-5:]:
            self.assertIn('--dependency=aftercorr:801', command)
        self.assertEqual(len((self.run / 'job_ids.txt').read_text().strip().split(',')), 11)

    def test_deadline_and_runtime_guards(self):
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            worker.budget_available({'deadline_utc': (now + timedelta(seconds=610)).isoformat()}, 600)
        self.assertGreater(worker.budget_available({'deadline_utc': (now + timedelta(seconds=800)).isoformat()}, 600), 790)
        manifest = {'threads': 4, 'software_at_freeze': {'python': 'fixture'}}
        with patch.object(worker, 'software', return_value={'python': 'changed'}), self.assertRaisesRegex(ValueError, 'versions'):
            worker.runtime(manifest)
        with patch.object(worker, 'software', return_value=manifest['software_at_freeze']), patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8'}):
            with self.assertRaisesRegex(ValueError, 'allocation'):
                worker.runtime(manifest)

    def test_bad_parent_prevents_all_evaluation_and_records_failure(self):
        manifest = self.prepare()
        with patch.object(worker, 'verify', return_value=manifest), patch.object(worker, 'runtime'), patch.object(worker, 'verified_training', side_effect=ValueError('parent incomplete')), patch.object(worker, 'evaluate_split') as evaluate:
            with self.assertRaisesRegex(ValueError, 'parent incomplete'):
                worker.run_job(self.run, 'grpo', 0)
        evaluate.assert_not_called()
        task = launch.find_job(manifest, 'grpo', 0)
        record = json.loads((launch.job_directory(self.run, task) / 'launch.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(launch.task_directory(self.run, manifest['evaluation_tasks'][task['evaluation_tasks'][0]]).exists())

    def test_out_of_scope_sequence_is_rejected_before_search(self):
        manifest = self.prepare()
        with patch.object(worker, 'verify', return_value=manifest), patch.object(worker, 'runtime'):
            with self.assertRaisesRegex(ValueError, 'outside'):
                worker.evaluate_one(self.run, 0, 'validation_000', self.root / 'result.json')
        self.assertFalse((self.root / 'result.json').exists())

    def test_failed_child_preserves_progress_and_never_claims_complete(self):
        manifest = self.prepare()
        task = next(t for t in manifest['evaluation_tasks'] if t['arm'] == 'cp_sat')
        with patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fixture', 720)):
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.evaluate_split(self.run, manifest, task, None)
        folder = launch.task_directory(self.run, task)
        saved = json.loads((folder / 'evaluation.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['results'], [])
        self.assertEqual(json.loads((folder / 'launch.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
