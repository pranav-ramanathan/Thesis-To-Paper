"""Small deterministic behavioral checks; no benchmark training or evaluation."""
import importlib.util
import random
import time
import unittest
from pathlib import Path

import torch

try:
    from . import geometry
    from .model import Policy, encode_states, sample, supervised_loss, trajectory_log_probs
except ImportError:
    import geometry
    from model import Policy, encode_states, sample, supervised_loss, trajectory_log_probs


class GeometryTests(unittest.TestCase):
    def test_contacts_and_invalid_witnesses(self):
        square = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        self.assertEqual(geometry.validate("HHHH", square), 1)
        self.assertEqual(geometry.validate("HPPH", square), 1)
        self.assertEqual(geometry.validate("PPPP", square), 0)
        self.assertEqual(sum(geometry.trajectory("HHHH", square)["rewards"]), 1)
        for bad in (square[:-1], square[:-1] + [square[0]],
                    square[:-1] + [[0.0, 1, 0]], square[:-1] + [[2, 2, 2]]):
            with self.assertRaises(ValueError):
                geometry.validate("HHHH", bad)

    def test_all_symmetries_deduplicate_and_augmentation_transforms_actions(self):
        fold = geometry.greedy("HHPHPPHHPPHPHH", random.Random(9))
        self.assertIsNotNone(fold)
        key = geometry.canonical_key(fold["positions"])
        for permutation, signs in geometry._SYMMETRIES:
            transformed = geometry._transform(fold["positions"], permutation, signs)
            self.assertEqual(geometry.canonical_key(transformed), key)
        for seed in range(48):
            augmented = geometry.augment(fold["sequence"], fold["positions"], random.Random(seed))
            self.assertEqual(augmented["contacts"], fold["contacts"])
            self.assertEqual(geometry.canonical_key(augmented["positions"]), key)
            rebuilt = [[0, 0, 0]]
            for action in augmented["actions"]:
                rebuilt.append([v + d for v, d in zip(rebuilt[-1], geometry.MOVES[action])])
            self.assertEqual(rebuilt, augmented["positions"])

    def test_independent_legacy_verifier_agrees_on_random_greedy_folds(self):
        path = Path(__file__).resolve().parents[1] / "cluster" / "geometry.py"
        spec = importlib.util.spec_from_file_location("independent_cluster_geometry", path)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        rng = random.Random(73)
        for length in (7, 13, 19, 27):
            for _ in range(3):
                sequence = "".join(rng.choice("HP") for _ in range(length))
                fold = geometry.greedy(sequence, rng)
                self.assertIsNotNone(fold)
                self.assertEqual(fold["contacts"], verifier.contacts(sequence, fold["positions"], length // 2))
                self.assertEqual(sum(fold["rewards"]), fold["contacts"])


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.policy = Policy({"width": 16, "heads": 2, "layers": 1})

    def test_padding_and_future_geometry_do_not_change_logits(self):
        seq = "HHPHPHH"
        prefix = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        single = encode_states([seq], [prefix], "cpu")
        batched = encode_states([seq, "HP" * 9], [prefix, prefix], "cpu")
        expected = self.policy(*single)
        torch.testing.assert_close(expected[0], self.policy(*batched)[0], atol=1e-6, rtol=1e-6)
        poisoned = list(single)
        poisoned[1] = poisoned[1].clone()
        poisoned[1][~poisoned[2]] = 100000
        torch.testing.assert_close(expected, self.policy(*poisoned))

    def test_samples_are_valid_reproducible_and_do_not_consume_global_rng(self):
        sequences = ["HHPHPHH", "HP" * 7, "PPPPPPP", "HH"]
        before = torch.random.get_rng_state().clone()
        a = sample(self.policy, sequences, torch.Generator().manual_seed(99))
        b = sample(self.policy, sequences, torch.Generator().manual_seed(99))
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(self.policy.training)
        for item in a:
            if item["complete"]:
                self.assertEqual(item["contacts"], geometry.validate(item["sequence"], item["positions"]))
            else:
                self.assertIsNone(item["contacts"])
            self.assertTrue(all(action in range(6) for action in item["actions"]))
        self.assertEqual(a[2]["contacts"], 0)

    def test_legal_mask_renormalizes_likelihood_and_gradient(self):
        # Force uniform raw logits: at the third placement exactly five moves
        # are legal because returning to the origin is forbidden.
        with torch.no_grad():
            self.policy.output.weight.zero_()
            self.policy.output.bias.zero_()
        item = geometry.trajectory("HHPH", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        logp, mask = trajectory_log_probs(self.policy, [item])
        self.assertEqual(mask.sum().item(), 2)
        torch.testing.assert_close(logp[0, 0].exp(), torch.tensor(0.2))
        loss = supervised_loss(self.policy, [item])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.policy.parameters() if p.grad is not None))
        self.assertGreater(self.policy.output.bias.grad.abs().sum().item(), 0)

    def test_teacher_forcing_is_invariant_to_batch_padding(self):
        a = geometry.greedy("HHPHPHH", random.Random(1))
        b = geometry.greedy("HPPH" * 4, random.Random(2))
        one, _ = trajectory_log_probs(self.policy, [a], chunk_size=1)
        both, masks = trajectory_log_probs(self.policy, [a, b], chunk_size=64)
        torch.testing.assert_close(one[0], both[0, masks[0]], atol=2e-6, rtol=2e-6)
        self.assertEqual(masks.sum(1).tolist(), [len(a["positions"]) - 2, len(b["positions"]) - 2])

    def test_traps_deadlines_and_prefixes_are_explicit(self):
        trapped = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                   [-1, 1, 1], [-1, 0, 1], [-1, -1, 1], [0, -1, 1], [0, -1, 2],
                   [0, 0, 2], [0, 0, 1]]
        result = sample(self.policy, ["H" * 13], prefixes=[trapped])[0]
        self.assertTrue(result["trapped"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["contacts"])
        self.assertEqual(result["positions"], trapped)
        result = sample(self.policy, ["HPHPHP"], deadline=time.monotonic() - 1)[0]
        self.assertTrue(result["interrupted"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["trapped"])
        self.assertIsNone(result["contacts"])
        self.assertEqual(len(result["positions"]), 2)

    def test_complete_anchor_only_loss_is_finite(self):
        item = geometry.trajectory("HH", [[0, 0, 0], [1, 0, 0]])
        loss = supervised_loss(self.policy, [item])
        self.assertEqual(loss.item(), 0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
