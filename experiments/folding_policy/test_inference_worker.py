"""Inference-only guardrails, partial output preservation and timed baselines."""
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

from dataset import make_dataset
import evaluate
import inference_worker as worker
from model import Policy
from train import state_hash


class InferenceWorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.addCleanup(patch.stopall)
        patch.object(worker.platform, 'processor', return_value='test CPU').start()
        patch.object(worker.platform, 'platform', return_value='test platform').start()
        patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '4'}).start()

    def fixture(self, run, method='greedy'):
        (run / 'tasks').mkdir()
        data = make_dataset()
        worker.write_json(run / 'data.json', data)
        task = {'task': 0, 'arm': method, 'seed': 0, 'method': method,
                'seconds': .02, 'checkpoint': None, 'checkpoint_sha256': None,
                'state_sha256': None, 'training_seconds': 0, 'slurm_seconds': 2000}
        manifest = {'threads': 4, 'software_at_freeze': worker.software(),
                    'hashes': {'data.json': worker.sha(run / 'data.json')}, 'tasks': [task],
                    'validation_ids': [row['id'] for row in data['validation']]}
        return manifest, task, data

    def checkpoint(self, run, manifest, task, arm='balanced'):
        policy = Policy({'width': 16, 'heads': 2, 'layers': 1, 'dropout': 0.0})
        state = copy.deepcopy(policy.state_dict())
        saved = {'schema': 1, 'kind': 'allocation_training_state', 'mode': 'rl_branch',
                 'condition': 'patternboost_rl', 'stage': 'rl', 'seed': 0, 'threads': 4,
                 'training_seconds': 7200, 'dataset_sha256': manifest['hashes']['data.json'],
                 'state_sha256': state_hash(state), 'model_config': policy.config,
                 'model_state': state, 'optimizer_state': {}, 'rng_states': {}, 'elite_pool': {},
                 'final_test_enabled': False, 'test_evaluated': False, 'selection': {'selected': False}}
        checkpoint = run / 'checkpoint.pt'
        torch.save(saved, checkpoint)
        task.update(arm=arm, method='model', checkpoint='checkpoint.pt', training_seconds=7200,
                    checkpoint_sha256=worker.sha(checkpoint), state_sha256=saved['state_sha256'])
        return saved

    def test_endpoint_identity_seed_stage_duration_and_weights_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _ = self.fixture(run)
            saved = self.checkpoint(run, manifest, task)
            metadata = worker.validate_endpoint(saved, manifest['hashes']['data.json'], task, 4)
            self.assertEqual(metadata['kind'], 'allocation_training_state')
            self.assertEqual(metadata['training_seconds'], 7200)
            for key, value in [('seed', 1), ('mode', 'patternboost'), ('training_seconds', 3600),
                               ('threads', 8), ('state_sha256', 'wrong'), ('test_evaluated', True)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.validate_endpoint({**saved, key: value}, manifest['hashes']['data.json'], task, 4)
            with self.assertRaisesRegex(ValueError, 'weights differ'):
                worker.validate_endpoint(saved, manifest['hashes']['data.json'],
                                         {**task, 'state_sha256': 'other-endpoint'}, 4)
            with self.assertRaisesRegex(ValueError, 'declared arm'):
                worker.validate_endpoint(saved, manifest['hashes']['data.json'],
                                         {**task, 'arm': 'patternboost'}, 4)

    def test_validation_only_greedy_inference_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, data = self.fixture(run)
            task['seconds'] = .2
            output = run / 'one.json'
            result = worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], output)
            row = result['result']
            self.assertGreater(row['generated_complete_folds'], 0)
            self.assertEqual(evaluate.verified_score(row['sequence'], row['best_positions']), row['best_contacts'])
            self.assertGreaterEqual(row['elapsed_s'], task['seconds'])
            self.assertIsNone(result['checkpoint_metadata'])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], output)
            with self.assertRaisesRegex(ValueError, 'validation partition'):
                worker.evaluate_one(run, task, manifest, data['test'][0]['id'], run / 'forbidden.json')
            self.assertFalse((run / 'forbidden.json').exists())

    def test_model_checkpoint_guard_is_called_by_real_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _ = self.fixture(run)
            self.checkpoint(run, manifest, task)
            task['seconds'] = .35
            result = worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], run / 'one.json')
            self.assertEqual(result['checkpoint_metadata']['state_sha256'], task['state_sha256'])
            self.assertEqual(result['result']['checkpoint_condition'], 'patternboost_rl')
            self.assertGreaterEqual(result['result']['elapsed_s'], task['seconds'])
            self.assertEqual(result['checkpoint_sha256'], task['checkpoint_sha256'])

    def test_checkpoint_mutation_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _ = self.fixture(run)
            self.checkpoint(run, manifest, task)
            (run / 'checkpoint.pt').write_text('changed')
            with patch.object(worker, 'evaluate_sequence') as inference:
                with self.assertRaisesRegex(ValueError, 'checkpoint file changed'):
                    worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], run / 'one.json')
            inference.assert_not_called()

    def test_real_cp_sat_search_uses_the_frozen_validation_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, data = self.fixture(run, 'cp_sat')
            task['seconds'] = 2
            result = worker.evaluate_one(run, task, manifest, manifest['validation_ids'][0], run / 'cp.json')
            row = result['result']
            self.assertEqual(row['sequence'], data['validation'][0]['sequence'])
            self.assertIn(row['solver_status'], ('NOT_STARTED', 'UNKNOWN', 'FEASIBLE', 'OPTIMAL'))
            self.assertIsNone(row['generated_complete_folds'])
            if row['best_positions'] is not None:
                self.assertEqual(evaluate.verified_score(row['sequence'], row['best_positions']), row['best_contacts'])

    def test_worker_keeps_finished_sequence_when_next_subprocess_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _ = self.fixture(run)
            manifest['validation_ids'] = manifest['validation_ids'][:2]
            calls = []

            def child(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    result = {'id': manifest['validation_ids'][0], 'method': 'greedy',
                              'seed': 0, 'budget_s': task['seconds'], 'elapsed_s': task['seconds']}
                    worker.write_json(Path(command[-1]), {'arm': 'greedy', 'checkpoint_sha256': None,
                                                          'checkpoint_metadata': None, 'result': result})
                    return Namespace(returncode=0)
                return Namespace(returncode=7)

            with patch.object(worker, 'verify', return_value=manifest), patch.object(worker.subprocess, 'run', side_effect=child):
                with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                    worker.execute(Namespace(directory=run, task=0))
            folder = worker.task_directory(run, task)
            payload = json.loads((folder / 'evaluation.json').read_text())
            launch = json.loads((folder / 'launch.json').read_text())
            self.assertEqual(payload['status'], 'failed')
            self.assertEqual(len(payload['results']), 1)
            self.assertEqual(launch['phases'][1]['exit_code'], 7)
            self.assertEqual(launch['status'], 'failed')
            self.assertTrue(payload['model_setup_charged_per_sequence'])
            self.assertTrue(all('train.py' not in str(command) for command in calls))
            with patch.object(worker, 'verify', return_value=manifest), self.assertRaises(FileExistsError):
                worker.execute(Namespace(directory=run, task=0))

    def test_wrong_resources_and_software_fail_before_starting_search(self):
        for drift in ('cpu', 'software'):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                manifest, _, _ = self.fixture(run)
                if drift == 'software':
                    manifest['software_at_freeze'] = {'other': 'version'}
                with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '8' if drift == 'cpu' else '4'}):
                    with patch.object(worker, 'verify', return_value=manifest), patch.object(worker.subprocess, 'run') as inference:
                        with self.assertRaisesRegex(ValueError, 'CPU allocation|package versions'):
                            worker.execute(Namespace(directory=run, task=0))
                    inference.assert_not_called()
                payload = json.loads((worker.task_directory(run, manifest['tasks'][0]) / 'evaluation.json').read_text())
                self.assertEqual(payload['status'], 'failed')
                self.assertEqual(payload['results'], [])

    def test_per_sequence_timeout_is_bounded_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest, task, _ = self.fixture(run)
            with patch.object(worker, 'verify', return_value=manifest):
                with patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 60)) as child:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        worker.execute(Namespace(directory=run, task=0))
            self.assertLessEqual(child.call_args.kwargs['timeout'], task['seconds'] + 60)
            launch = json.loads((worker.task_directory(run, task) / 'launch.json').read_text())
            self.assertTrue(launch['phases'][0]['timed_out'])
            self.assertEqual(launch['status'], 'failed')

    def test_cp_proof_keeps_actual_short_elapsed_and_full_domain_scope(self):
        positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]

        def solver(sequence, **kwargs):
            kwargs['emit']({'type': 'witness', 'positions': positions, 'contacts': 1})
            kwargs['emit']({'type': 'solver_end', 'status': 'OPTIMAL', 'contacts': 1,
                            'bound': 1, 'scope': 'full_declared_cube'})

        with patch.object(evaluate, 'load_cp_solver', return_value=solver):
            result = evaluate.evaluate_sequence({'id': 'small', 'sequence': 'HHHH'}, 'cp_sat', 0, 60, 4)
        self.assertTrue(result['proof_within_budget'])
        self.assertLess(result['elapsed_s'], 60)
        self.assertEqual(result['bound_scope'], 'full_declared_cube')
        self.assertEqual(result['best_contacts'], 1)
        self.assertIsNone(result['generated_complete_folds'])


if __name__ == '__main__':
    unittest.main()
