"""Small numerical/semantic checks; no final benchmark sequence is evaluated."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

try:
    from . import geometry
    from .train import (ElitePool, check_splits, greedy_complete, group_advantages, stratified_rows,
                        outcome, state_hash, train)
except ImportError:
    import geometry
    from train import (ElitePool, check_splits, greedy_complete, group_advantages, stratified_rows,
                       outcome, state_hash, train)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None


class SemanticsTest(unittest.TestCase):
    def test_group_normalization_and_constant_groups(self):
        self.assertEqual(group_advantages([3, 3, 3]), [0, 0, 0])
        values = group_advantages([-10, 0, 2, 5])
        self.assertAlmostEqual(sum(values), 0)
        self.assertAlmostEqual(sum(v * v for v in values) / len(values), 1)
        self.assertLess(values[0], values[-1])

    def test_trap_return_is_negative_even_after_many_contacts(self):
        failed = {"complete": False, "interrupted": False, "rewards": [40], "contacts": None}
        self.assertEqual(outcome(failed, -10), -10)
        failed["interrupted"] = True
        self.assertIsNone(outcome(failed, -10))

    def test_reversed_sequence_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_splits({"train": [{"sequence": "HHPPHP"}],
                          "validation": [{"sequence": "PHPPHH"}]})

    def test_bounded_data_order_covers_length_and_composition(self):
        rows = [{"sequence": seq} for seq in ["HHHP", "HHPH", "PPPH", "PPHP", "HHHHHP", "PHHHHH"]]
        ordered = stratified_rows(rows, random.Random(4))
        first_strata = {(len(row["sequence"]), row["sequence"].count("H")) for row in ordered[:3]}
        self.assertEqual(first_strata, {(4, 3), (4, 1), (6, 5)})

    def test_symmetry_duplicates_do_not_grow_elite_pool(self):
        seq = "HHHHHHHH"
        fold = geometry.greedy(seq, rng=random.Random(2))
        pool = ElitePool(4, geometry)
        self.assertTrue(pool.add(fold))
        reflected = geometry.augment(seq, fold["positions"], random.Random(5))
        self.assertFalse(pool.add(reflected))
        self.assertEqual(len(pool), 1)

    def test_greedy_local_completion_preserves_ordered_prefix(self):
        seq = "HHHPPHHHPPHH"
        original = geometry.greedy(seq, rng=random.Random(3))
        prefix = original["positions"][:5]
        completed = greedy_complete(seq, prefix, random.Random(3), geometry, time.monotonic() + 1)
        self.assertEqual(completed["positions"][:5], prefix)
        self.assertTrue(completed["complete"])
        self.assertEqual(geometry.validate(seq, completed["positions"]), completed["contacts"])


@unittest.skipUnless(HAS_TORCH, "Torch is optional for dependency-free geometry checks")
class NumericalTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.torch = torch
        try:
            from .model import Policy, sample, supervised_loss, trajectory_log_probs
        except ImportError:
            from model import Policy, sample, supervised_loss, trajectory_log_probs
        self.Policy, self.sample = Policy, sample
        self.supervised_loss, self.log_probs = supervised_loss, trajectory_log_probs

    def test_supervised_fit_reduces_loss(self):
        model = self.Policy({"width": 16, "heads": 2, "layers": 1})
        rows = [geometry.greedy("HHHPPHHHHH", rng=random.Random(i)) for i in range(3)]
        optimizer = self.torch.optim.Adam(model.parameters(), lr=0.01)
        first = float(self.supervised_loss(model, rows).detach())
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            loss = self.supervised_loss(model, rows)
            loss.backward()
            optimizer.step()
        last = float(self.supervised_loss(model, rows).detach())
        self.assertLess(last, first * 0.9)

    def test_grpo_objective_produces_finite_nonzero_update(self):
        torch = self.torch
        model = self.Policy({"width": 16, "heads": 2, "layers": 1})
        rows = self.sample(model, ["HHHHHHHHHHHH"] * 16,
                           generator=torch.Generator().manual_seed(21))
        rewards = [outcome(row, -10) for row in rows]
        advantages = group_advantages(rewards)
        self.assertTrue(any(advantages), "Need a varied group to test a policy update")
        with torch.no_grad():
            old, mask = self.log_probs(model, rows)
        before = state_hash(model.state_dict())
        current, taken = self.log_probs(model, rows)
        self.assertTrue(torch.equal(mask, taken))
        ratio = (current - old).exp()
        advantage = torch.tensor(advantages).unsqueeze(1)
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
        loss = -((surrogate * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertNotEqual(before, state_hash(model.state_dict()))

    def _combined_smoke(self, condition):
        dataset = {"train": [{"sequence": "HHHHHHHH"}, {"sequence": "HHPPHHHH"}],
                   "validation": [{"sequence": "HHHPHHHH"}],
                   "test": [{"sequence": "HHPHPHHH"}]}
        config = {"model": {"width": 16, "heads": 2, "layers": 1},
                  "training": {"batch_size": 4, "group_size": 8, "cp_seconds": 0.4,
                               "validation_fraction": 0.2, "data_fraction": 0.5,
                               "validation_folds_per_sequence": 2, "elite_size": 2}}
        with tempfile.TemporaryDirectory(prefix="hp-training-test-") as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps(dataset))
            result = train(path, condition, 7, 12 if condition == "cp_rl" else 8, 1, Path(folder) / "run", config)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["test_evaluated"])
            self.assertGreater(result["counters"]["supervised_updates"], 0)
            self.assertGreater(result["counters"]["rl_updates"], 0)
            handoff = result["handoff"]
            self.assertEqual(handoff["pretraining_sha256"], handoff["rl_initial_sha256"])
            self.assertTrue(handoff["same_model_object"])
            self.assertTrue(handoff["optimizer_state_retained"])
            self.assertNotEqual(handoff["pretraining_sha256"], result["final_state_sha256"])
            self.assertTrue(result["selection"]["validated"])
            self.assertTrue((Path(folder) / "run" / "checkpoint.pt").exists())

    def test_patternboost_handoff_smoke(self):
        self._combined_smoke("patternboost_rl")

    @unittest.skipUnless(HAS_ORTOOLS, "CP data generation needs OR-Tools")
    def test_cp_handoff_smoke(self):
        self._combined_smoke("cp_rl")


if __name__ == "__main__":
    unittest.main()
