"""Task isolation, bounded failures and real standalone parent inference."""
from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from allocation_train import AllocationTrainer
from dataset import make_dataset
from evaluate import verified_score
import finetune_worker as worker


class FinetuneWorkerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}).start()
        patch.object(worker.platform, 'processor', return_value='test CPU').start()
        patch.object(worker.platform, 'platform', return_value='test platform').start()

    def fixture(self, run):
        (run / 'tasks').mkdir()
        data = make_dataset()
        worker.write_json(run / 'data.json', data)
        config = {'model': {'width': 16, 'heads': 2, 'layers': 1, 'dropout': 0.0}}
        trainer = AllocationTrainer(run / 'data.json', 0, 1, config)
        trainer.training_seconds = 7200
        saved = trainer.checkpoint()
        torch.save(saved, run / 'parent.pt')
        task = {'task': 15, 'arm': 'parent', 'seed': 0, 'parent': 'parent.pt',
                'parent_sha256': worker.sha(run / 'parent.pt'),
                'parent_state_sha256': saved['state_sha256'],
                'parent_optimizer_sha256': saved['optimizer_sha256'],
                'training_seconds': 0, 'lineage_training_seconds': 7200,
                'learning_rate': 0, 'evaluation_seconds': .3, 'slurm_seconds': 4500}
        manifest = {'threads': 1, 'software_at_freeze': worker.software(),
                    'hashes': {'data.json': worker.sha(run / 'data.json')}, 'tasks': [task],
                    'parent_training_seconds': 7200,
                    'validation_ids': [row['id'] for row in data['validation']]}
        return manifest, task, data, saved

    def test_real_parent_inference_has_no_training_and_verifies_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, saved = self.fixture(run)
            output = run / 'one.json'
            result = worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], output)
            self.assertEqual(result['checkpoint_metadata']['state_sha256'], saved['state_sha256'])
            self.assertGreaterEqual(result['result']['elapsed_s'], task['evaluation_seconds'])
            row = result['result']
            self.assertGreater(row['generated_complete_folds'], 0)
            self.assertEqual(verified_score(row['sequence'], row['best_positions']), row['best_contacts'])
            self.assertEqual(worker.sha(run / 'parent.pt'), task['parent_sha256'])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], output)

    def test_final_test_and_parent_mutation_rejected_before_search(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, data, _ = self.fixture(run)
            with patch.object(worker, 'evaluate_sequence') as search:
                with self.assertRaisesRegex(ValueError, 'validation partition'):
                    worker.evaluate_one(run, task, manifest, data['test'][0]['id'], run / 'forbidden.json')
                (run / 'parent.pt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Frozen parent changed'):
                    worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], run / 'one.json')
                search.assert_not_called()

    def test_parent_identity_is_bound_to_seed_state_duration_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, saved = self.fixture(run)
            for key, value in [('seed', 1), ('training_seconds', 3600), ('threads', 8),
                               ('test_evaluated', True), ('mode', 'rl_scratch')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.validate_endpoint({**saved, key: value}, manifest['hashes']['data.json'],
                                             task, manifest, saved['state_sha256'])
            with self.assertRaises(ValueError):
                worker.validate_endpoint(saved, manifest['hashes']['data.json'], task, manifest, 'wrong')

    def test_failed_sequence_keeps_previous_result_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, _ = self.fixture(run)
            manifest['validation_ids'] = manifest['validation_ids'][:2]
            calls = []

            def child(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    worker.write_json(Path(command[-1]), {
                        'arm': 'parent', 'checkpoint_sha256': task['parent_sha256'],
                        'checkpoint_metadata': {'state_sha256': task['parent_state_sha256']},
                        'result': {'id': manifest['validation_ids'][0], 'method': 'model',
                                   'seed': 0, 'budget_s': task['evaluation_seconds']}})
                    return Namespace(returncode=0)
                return Namespace(returncode=9)

            with patch.object(worker, 'verify', return_value=manifest), patch.object(worker.subprocess, 'run', side_effect=child):
                with self.assertRaisesRegex(RuntimeError, 'exited 9'):
                    worker.execute(Namespace(directory=run, task=15))
            folder = worker.task_directory(run, task)
            payload = json.loads((folder / 'evaluation.json').read_text())
            launch = json.loads((folder / 'launch.json').read_text())
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(len(payload['results']), 1)
            self.assertEqual(launch['phases'][1]['exit_code'], 9)
            self.assertTrue(all('finetune_train.py' not in ' '.join(command) for command in calls))
            with patch.object(worker, 'verify', return_value=manifest), self.assertRaises(FileExistsError):
                worker.execute(Namespace(directory=run, task=15))

    def test_subprocess_timeout_is_saved_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, _ = self.fixture(run)
            with patch.object(worker, 'verify', return_value=manifest):
                with patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sequence', 60)) as child:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        worker.execute(Namespace(directory=run, task=15))
            self.assertLessEqual(child.call_args.kwargs['timeout'], task['evaluation_seconds'] + 60)
            launch = json.loads((worker.task_directory(run, task) / 'launch.json').read_text())
            self.assertTrue(launch['phases'][0]['timed_out'])
            self.assertEqual(launch['status'], 'failed')

    def test_runtime_drift_cannot_start_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, _ = self.fixture(run)
            manifest['software_at_freeze'] = {'wrong': 'version'}
            with patch.object(worker, 'verify', return_value=manifest), patch.object(worker.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'package versions'):
                    worker.execute(Namespace(directory=run, task=15))
                child.assert_not_called()
            launch = json.loads((worker.task_directory(run, task) / 'launch.json').read_text())
            self.assertEqual(launch['status'], 'failed')

    def test_incomplete_training_cannot_reach_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _, _ = self.fixture(run)
            task.update(arm='grpo', training_seconds=1800, lineage_training_seconds=9000, learning_rate=3e-4)

            def child(command, **kwargs):
                folder = Path(command[-1])
                folder.mkdir()
                worker.write_json(folder / 'result.json', {'status': 'insufficient_training'})
                return Namespace(returncode=0)

            with patch.object(worker, 'verify', return_value=manifest), patch.object(worker.subprocess, 'run', side_effect=child) as proc:
                with self.assertRaisesRegex(ValueError, 'Training receipt'):
                    worker.execute(Namespace(directory=run, task=15))
            self.assertEqual(proc.call_count, 1)
            payload = json.loads((worker.task_directory(run, task) / 'evaluation.json').read_text())
            self.assertEqual(payload['results'], [])
            self.assertEqual(payload['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
