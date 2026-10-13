import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from allocation_evaluate import evaluate_endpoint, validate_allocation_checkpoint
from dataset import make_dataset
from evaluate import validate_checkpoint, verified_score
from model import Policy
from train import state_hash


class AllocationEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.policy = Policy({'width': 16, 'heads': 2, 'layers': 1, 'dropout': 0.0})
        state = copy.deepcopy(self.policy.state_dict())
        self.saved = {
            'schema': 1, 'kind': 'allocation_training_state',
            'mode': 'rl_branch', 'condition': 'patternboost_rl', 'stage': 'rl',
            'seed': 7, 'training_seconds': 7200, 'dataset_sha256': 'dataset',
            'state_sha256': state_hash(state), 'model_config': self.policy.config,
            'model_state': state, 'optimizer_state': {}, 'rng_states': {}, 'elite_pool': {},
            'final_test_enabled': False, 'test_evaluated': False,
            'selection': {'selected': False},
        }

    def test_original_selected_checkpoint_guard_is_preserved(self):
        validate_allocation_checkpoint(self.saved, 'dataset', 7)
        with self.assertRaisesRegex(ValueError, 'validation selection'):
            validate_checkpoint(self.saved, 'dataset')
        original = {**self.saved, 'kind': 'original_training_checkpoint',
                    'selection': {'split': 'validation', 'validated': True}}
        validate_checkpoint(original, 'dataset')
        with self.assertRaisesRegex(ValueError, 'allocation training state'):
            validate_allocation_checkpoint(original, 'dataset', 7)

    def test_allocation_guard_rejects_wrong_provenance_and_modified_weights(self):
        for key, value in [('dataset_sha256', 'other'), ('seed', 3), ('stage', 'patternboost'),
                           ('training_seconds', float('nan')), ('state_sha256', 'wrong'),
                           ('test_evaluated', True), ('selection', {'selected': True})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_allocation_checkpoint({**self.saved, key: value}, 'dataset', 7)
        saved = copy.deepcopy(self.saved)
        next(iter(saved['model_state'].values())).add_(1)
        with self.assertRaisesRegex(ValueError, 'parameter hash'):
            validate_allocation_checkpoint(saved, 'dataset', 7)

    def test_standalone_endpoint_evaluation_and_test_split_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = make_dataset()
            path = root / 'data.json'
            path.write_text(json.dumps(data))
            self.saved['dataset_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = root / 'checkpoint.pt'
            torch.save(self.saved, checkpoint)
            output = root / 'evaluation.json'
            result = evaluate_endpoint(path, checkpoint, 7, 0.5, 1, output,
                                       [data['validation'][0]['id']])
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['selection'], 'fixed_training_endpoint')
            self.assertFalse(result['final_test_access'])
            row = result['results'][0]
            self.assertGreater(row['generated_complete_folds'], 0)
            self.assertEqual(verified_score(row['sequence'], row['best_positions']), row['best_contacts'])
            self.assertGreaterEqual(row['elapsed_s'], row['budget_s'])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                evaluate_endpoint(path, checkpoint, 7, .5, 1, output, [data['validation'][0]['id']])
            with self.assertRaisesRegex(ValueError, 'validation partition'):
                evaluate_endpoint(path, checkpoint, 7, .5, 1, root / 'forbidden.json',
                                  [data['test'][0]['id']])
            self.assertFalse((root / 'forbidden.json').exists())


if __name__ == '__main__':
    unittest.main()
