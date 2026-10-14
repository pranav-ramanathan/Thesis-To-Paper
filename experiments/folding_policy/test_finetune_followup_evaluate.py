"""Attempt accounting includes traps, incomplete prefixes and charged deadlines."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from model import Policy
import finetune_followup_evaluate as evaluator


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FollowupEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / 'checkpoint.pt'
        config = {'width': 16, 'heads': 2, 'layers': 1, 'dropout': 0.0}
        policy = Policy(config)
        torch.save({'model_config': config, 'model_state': policy.state_dict(), 'seed': 3,
                    'stage': 'patternboost', 'dataset_sha256': 'data'}, self.checkpoint)
        self.row = {'id': 'validation_000', 'sequence': 'HHHH'}
        self.square = {'complete': True, 'positions': [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                       'contacts': 1, 'interrupted': False}
        self.straight = {'complete': True, 'positions': [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)],
                         'contacts': 0, 'interrupted': False}
        self.trap = {'complete': False, 'interrupted': False}
        self.interrupted = {'complete': False, 'interrupted': True}
        self.clock = FakeClock()

    def run_draw(self, draw, targets=(2, 4, 8, 16), **kwargs):
        with patch.object(evaluator.time, 'monotonic', self.clock), patch('model.sample', side_effect=draw):
            return evaluator.evaluate_sequence(self.row, 'model', 3, 1.0, 1, self.checkpoint,
                    data_hash='data', checkpoint_validator=lambda s, h: None,
                    attempt_checkpoints=targets, **kwargs)

    def test_ordered_prefix_includes_traps_and_preserves_zero_score(self):
        calls = []
        def draw(policy, sequences, generator, deadline):
            calls.append((len(sequences), generator.initial_seed(), deadline, policy.training))
            if len(calls) == 1:
                self.clock.value = .2
                return [self.straight, self.trap, self.square, self.trap] * 2
            self.clock.value = 1.1
            return [self.interrupted] * 8
        result = self.run_draw(draw)
        self.assertEqual(calls[0], (8, 3, 1.0, False))
        self.assertEqual(result['attempted_folds'], 16)
        self.assertEqual(result['generated_complete_folds'], 4)
        self.assertEqual(result['trapped_folds'], 4)
        self.assertEqual(result['interrupted_folds'], 8)
        self.assertEqual(result['attempt_prefixes'][0]['best_contacts'], 0)
        self.assertEqual(result['attempt_prefixes'][0]['trapped_folds'], 1)
        self.assertEqual(result['attempt_prefixes'][1]['best_contacts'], 1)
        self.assertFalse(result['attempt_prefixes'][-1]['covered'])
        self.assertIsNone(result['attempt_prefixes'][-1]['best_contacts'])
        self.assertEqual([p['attempt_index'] for p in result['curve']], [1, 3])
        for point in result['curve']:
            record = result['ordered_attempts'][point['attempt_index'] - 1]
            self.assertEqual((point['contacts'], point['seconds'], point['positions']),
                             (record['contacts'], record['seconds'], record['positions']))

    def test_unreached_prefix_is_missing_even_with_valid_folds(self):
        def draw(*args, **kwargs):
            self.clock.value += .6
            return [self.square] * 8
        result = self.run_draw(draw, targets=(8, 32))
        self.assertTrue(result['attempt_prefixes'][0]['covered'])
        prefix = result['attempt_prefixes'][1]
        self.assertFalse(prefix['reached'])
        self.assertEqual(prefix['observed_attempts'], 16)
        self.assertIsNone(prefix['best_contacts'])
        self.assertEqual(result['discarded_after_budget'], 8)
        self.assertTrue(all(p['seconds'] <= 1 for p in result['curve']))

    def test_verification_time_is_charged_and_late_witness_is_excluded(self):
        def draw(*args, **kwargs):
            self.clock.value = .99
            return [self.square] * 8
        original = evaluator.verified_score
        def slow_verify(*args):
            self.clock.value += .02
            return original(*args)
        with patch.object(evaluator, 'verified_score', side_effect=slow_verify):
            result = self.run_draw(draw, targets=(8,))
        self.assertEqual(result['generated_complete_folds'], 0)
        self.assertEqual(result['discarded_after_budget'], 8)
        self.assertIsNone(result['best_contacts'])
        self.assertFalse(result['attempt_prefixes'][0]['covered'])

    def test_no_sampling_if_loading_uses_entire_budget(self):
        def validate(*args):
            self.clock.value = 1.2
        with patch.object(evaluator.time, 'monotonic', self.clock), patch('model.sample') as sample:
            result = evaluator.evaluate_sequence(self.row, 'model', 3, 1.0, 1, self.checkpoint,
                      checkpoint_validator=validate, attempt_checkpoints=(8,))
        sample.assert_not_called()
        self.assertEqual(result['attempted_folds'], 0)
        self.assertGreater(result['setup_s'], 1)
        self.assertFalse(result['attempt_prefixes'][0]['covered'])

    def test_bad_reward_fails_and_saves_partial_progress(self):
        snapshots = []
        bad = {**self.square, 'contacts': 100}
        def draw(*args, **kwargs):
            self.clock.value = .2
            return [bad] * 8
        with self.assertRaisesRegex(ValueError, 'independent verifier'):
            self.run_draw(draw, progress=lambda result: snapshots.append(copy.deepcopy(result)))
        self.assertEqual(snapshots[-1]['status'], 'failed')
        self.assertEqual(snapshots[-1]['curve'], [])

    def test_real_sampler_keeps_checkpoint_unchanged_and_reports_cpu_resources(self):
        original = self.checkpoint.read_bytes()
        result = evaluator.evaluate_sequence(self.row, 'model', 3, .15, 1, self.checkpoint,
                 checkpoint_validator=lambda s, h: None, attempt_checkpoints=(8, 16))
        self.assertEqual(self.checkpoint.read_bytes(), original)
        self.assertEqual(result['status'], 'complete')
        self.assertGreater(result['generated_complete_folds'], 0)
        self.assertEqual(result['resource_usage']['threads'], 1)
        self.assertGreater(result['resource_usage']['process_cpu_s'], 0)
        self.assertEqual(len(result['ordered_attempts']), min(16, result['attempted_folds']))
        for row in result['ordered_attempts']:
            if row['status'] == 'complete':
                self.assertEqual(evaluator.verified_score('HHHH', row['positions']), row['contacts'])


if __name__ == '__main__':
    unittest.main()
