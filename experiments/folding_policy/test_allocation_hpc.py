"""Bounded checks for frozen allocation plans and dependent Slurm arrays."""
from argparse import Namespace
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import allocation_launch as launch
from dataset import make_dataset


class AllocationHpcTests(unittest.TestCase):
    def setUp(self):
        # platform.processor() can invoke subprocess internally on macOS; isolate
        # that discovery from these tests' process-launch mocks.
        self.addCleanup(patch.stopall)
        patch.object(launch.platform, 'processor', return_value='test CPU').start()
        patch.object(launch.platform, 'platform', return_value='test platform').start()

    def fixture(self, run):
        policy = run / 'bundle' / 'folding_policy'
        policy.mkdir(parents=True)
        (run / 'tasks').mkdir()
        (run / 'logs').mkdir()
        shutil.copy2(launch.HERE / 'allocation_protocol.json', policy / 'allocation_protocol.json')
        data = make_dataset()
        launch.write_json(run / 'data.json', data)
        relative = 'bundle/folding_policy/allocation_protocol.json'
        manifest = {
            'kind': 'validation_allocation_pilot', 'final_test_access': False,
            'no_automatic_followup': True, 'tasks': launch.tasks(), 'threads': 4,
            'software_at_freeze': launch.software(),
            'evaluation_seconds_per_sequence_per_seed': 60,
            'validation_ids': launch.validation_ids(data),
            'hashes': {'data.json': launch.sha(run / 'data.json'), relative: launch.sha(run / relative)},
        }
        self.save_manifest(run, manifest)
        return manifest

    def save_manifest(self, run, manifest):
        launch.write_json(run / 'allocation.json', manifest)
        (run / 'allocation.sha256').write_text(launch.sha(run / 'allocation.json') + '\n')

    def test_plan_has_twelve_logical_outcomes_and_full_validation(self):
        tasks = launch.tasks()
        self.assertEqual(len(tasks), 12)
        self.assertEqual(sum(row['training_seconds'] for row in tasks) * 4 / 3600, 66)
        self.assertEqual(sum(row['slurm_seconds'] for row in tasks) * 4 / 3600, 102)
        self.assertEqual(len(launch.validation_ids(make_dataset())), 16)
        self.assertEqual(sum(len(row['evaluation_stages']) for row in tasks), 18)
        for row in tasks:
            self.assertEqual(row['lineage_training_seconds'], 7200)
            if row['arm'] == 'balanced':
                self.assertEqual(row['resume_stage'], 'stage_3600')
            if row['arm'] == 'more_pretrain':
                self.assertEqual(row['resume_stage'], 'stage_5400')

    def test_manifest_and_frozen_source_mutations_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = self.fixture(run)
            launch.verify(run)
            (run / 'allocation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                launch.verify(run)
            self.save_manifest(run, manifest)
            (run / 'data.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                launch.verify(run)

    def test_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            manifest = self.fixture(run)
            external = Path(directory) / 'external.txt'
            external.write_text('do not read as a run artifact')
            manifest['hashes']['../external.txt'] = launch.sha(external)
            self.save_manifest(run, manifest)
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                launch.verify(run)

    def test_array_dependencies_and_ids_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            responses = [Namespace(stdout=f'{number}\n', stderr='A scheduler warning\n' if number == 101 else '')
                         for number in (101, 102, 103, 104)]
            with patch.object(launch.subprocess, 'run', side_effect=responses) as submit:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = launch.submit_arrays(run)
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['arrays'][0]['sbatch_stderr'], 'A scheduler warning\n')
            commands = [call.args[0] for call in submit.call_args_list]
            self.assertFalse(any(arg.startswith('--dependency') for arg in commands[0]))
            self.assertFalse(any(arg.startswith('--dependency') for arg in commands[1]))
            self.assertIn('--dependency=aftercorr:101', commands[2])
            self.assertIn('--dependency=aftercorr:101', commands[3])
            self.assertEqual((run / 'job_ids.txt').read_text().strip(), '101,102,103,104')
            self.assertEqual([command[-1] for command in commands], list(launch.ARMS))

    def test_partial_submission_keeps_accepted_jobs_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            error = subprocess.CalledProcessError(1, ['sbatch'], output='job submit limit reached')
            with patch.object(launch.subprocess, 'run', side_effect=[Namespace(stdout='201\n', stderr=''), error]):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'accepted arrays remain scheduled: 201'):
                    launch.submit_arrays(run)
            record = json.loads((run / 'submission.json').read_text())
            self.assertEqual(record['status'], 'partial_submission')
            self.assertTrue(record['arrays'][0]['submitted'])
            self.assertEqual(record['arrays'][1]['sbatch_output'], 'job submit limit reached')
            self.assertEqual((run / 'job_ids.txt').read_text().strip(), '201')

    def test_dry_run_never_calls_sbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(launch.subprocess, 'run') as submit:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = launch.submit_arrays(Path(directory), dry_run=True)
            submit.assert_not_called()
            self.assertEqual(result['status'], 'prepared')
            self.assertEqual(result['arrays'][2]['dependency'], 'aftercorr:PARENT_ARRAY_JOB_ID')

    def test_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                launch.prepare(Namespace(directory=directory, python='/unused', dry_run=True))

    def test_protocol_cannot_silently_change_the_learning_algorithm(self):
        config = json.loads((launch.HERE / 'allocation_protocol.json').read_text())
        launch.validate_protocol(config)
        config['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'Training objective settings'):
            launch.validate_protocol(config)

    def test_runtime_cpu_mismatch_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture(run)
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8'}):
                with self.assertRaisesRegex(ValueError, 'CPU allocation'):
                    launch.execute(Namespace(directory=str(run), arm='rl_scratch', seed=0))
            result = json.loads((run / 'tasks' / 'rl_scratch_seed0' / 'launch.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['phases'], [])

    def test_branch_requires_successful_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture(run)
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '4'}):
                with self.assertRaisesRegex(ValueError, 'parent did not finish'):
                    launch.execute(Namespace(directory=str(run), arm='balanced', seed=0))
            result = json.loads((run / 'tasks' / 'balanced_seed0' / 'launch.json').read_text())
            self.assertEqual(result['status'], 'failed')

    def test_subprocess_failure_preserves_phase_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture(run)
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '4'}):
                with patch.object(launch.subprocess, 'run', return_value=Namespace(returncode=7)) as worker:
                    with self.assertRaisesRegex(RuntimeError, 'training exited 7'):
                        launch.execute(Namespace(directory=str(run), arm='rl_scratch', seed=0))
            self.assertEqual(worker.call_count, 1)
            result = json.loads((run / 'tasks' / 'rl_scratch_seed0' / 'launch.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['phases'][0]['exit_code'], 7)

    def test_branch_rejects_mutated_parent_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.fixture(run)
            parent = run / 'tasks' / 'patternboost_seed0'
            (parent / 'training').mkdir(parents=True)
            launch.write_json(parent / 'launch.json', {'status': 'complete'})
            checkpoint = parent / 'training' / 'stage_3600.pt'
            checkpoint.write_text('original')
            launch.write_json(parent / 'training' / 'result.json', {
                'status': 'complete', 'checkpoint_records': [
                    {'filename': checkpoint.name, 'file_sha256': launch.sha(checkpoint)}]})
            checkpoint.write_text('changed')
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '4'}):
                with patch.object(launch.subprocess, 'run') as worker:
                    with self.assertRaisesRegex(ValueError, 'Parent checkpoint differs'):
                        launch.execute(Namespace(directory=str(run), arm='balanced', seed=0))
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
