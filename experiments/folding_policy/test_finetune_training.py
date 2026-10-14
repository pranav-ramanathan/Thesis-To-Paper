"""Disposable objective, handoff, cutoff and provenance checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

try:
    from . import geometry
    from .allocation_train import AllocationTrainer, file_sha256, nested_hash
    from .finetune_train import (ARMS, FineTuneTrainer, dpo_loss, dr_grpo_loss,
                                greedy_policy_rollout, preference_pairs, reinforce_loss,
                                run, validate_finetune_checkpoint)
    from .train import outcome, state_hash
except ImportError:
    import geometry
    from allocation_train import AllocationTrainer, file_sha256, nested_hash
    from finetune_train import (ARMS, FineTuneTrainer, dpo_loss, dr_grpo_loss,
                               greedy_policy_rollout, preference_pairs, reinforce_loss,
                               run, validate_finetune_checkpoint)
    from train import outcome, state_hash


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class FineTuneTrainingTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory(prefix="hp-finetune-training-")
        self.folder = Path(self.temp.name)
        self.data = self.folder / "data.json"
        self.data.write_text(json.dumps({"train": [{"sequence": "HHHHHHHH"}],
                                        "validation": [{"sequence": "HHHPHHHH"}],
                                        "test": [{"sequence": "HHPHPHHH"}]}))
        self.config = {"model": {"width": 16, "heads": 2, "layers": 1, "dropout": 0.0},
                       "training": {"batch_size": 4, "group_size": 8, "elite_size": 2}}
        self.parent = AllocationTrainer(self.data, 7, 1, config=self.config)
        self.parent.initialize_examples(time.monotonic() + 0.2)
        self.parent.supervised_step()
        self.parent.training_seconds = 7200.0
        self.parent.measured_training_seconds = 7200.0
        self.saved = self.parent.checkpoint()
        self.path = self.folder / "parent.pt"
        torch.save(self.saved, self.path)
        low = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
               (4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 3, 0)]
        high = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)]
        self.low = geometry.trajectory("HHHHHHHH", low)
        self.high = geometry.trajectory("HHHHHHHH", high)

    def tearDown(self):
        self.temp.cleanup()

    def trainer(self, arm):
        return FineTuneTrainer(self.data, self.path, arm, 7, 1)

    def test_dr_grpo_uses_global_length_and_unnormalized_advantage(self):
        torch = self.torch
        current = torch.zeros((2, 3), requires_grad=True)
        taken = torch.tensor([[True, True, True], [True, False, False]])
        loss = dr_grpo_loss(current, current.detach().clone(), taken, [2.0, -2.0], 10)
        self.assertAlmostEqual(float(loss.detach()), -0.2)
        loss.backward()
        expected = torch.tensor([[-0.1, -0.1, -0.1], [0.1, 0.0, 0.0]])
        torch.testing.assert_close(current.grad, expected)
        other = torch.zeros((2, 3), requires_grad=True)
        dr_grpo_loss(other, other.detach().clone(), taken, [4.0, -4.0], 10).backward()
        torch.testing.assert_close(other.grad, 2 * expected)

    def test_reinforce_sequence_sum_preserves_trap_penalty(self):
        torch = self.torch
        values = torch.zeros((2, 3), requires_grad=True)
        taken = torch.tensor([[True, True, True], [True, False, False]])
        reinforce_loss(values, taken, [5 - 2, -10 - 2], 10).backward()
        torch.testing.assert_close(values.grad, torch.tensor([[-0.15, -0.15, -0.15], [0.6, 0., 0.]]))
        partial = geometry.trajectory("HHHHHHHH", [(0, 0, 0), (1, 0, 0)])
        self.assertEqual(outcome(partial, -10), -10)
        partial["interrupted"] = True
        self.assertIsNone(outcome(partial, -10))

    def test_dpo_gradient_increases_preferred_relative_likelihood(self):
        torch = self.torch
        current = torch.tensor([[-2.0, -3.0]], requires_grad=True)
        reference = current.detach().clone().requires_grad_(True)
        loss = dpo_loss(current, reference, beta=0.1)
        self.assertAlmostEqual(float(loss.detach()), 0.69314718, places=6)
        loss.backward()
        self.assertLess(float(current.grad[0, 0]), 0)
        self.assertGreater(float(current.grad[0, 1]), 0)
        self.assertIsNone(reference.grad)

    def test_dpo_pairs_independently_reject_invalid_incomplete_interrupted_and_ties(self):
        bad = copy.deepcopy(self.high)
        bad["contacts"] += 1
        interrupted = {**self.high, "interrupted": True}
        partial = geometry.trajectory("HHHHHHHH", [(0, 0, 0), (1, 0, 0)])
        pairs = preference_pairs([bad, interrupted, partial, self.low, self.high])
        self.assertEqual(pairs, [(self.high, self.low)])
        self.assertEqual(preference_pairs([self.high, copy.deepcopy(self.high)]), [])
        other = {**self.high, "sequence": "HHPPHHHH"}
        with self.assertRaisesRegex(ValueError, "one input"):
            preference_pairs([self.high, other])

    def test_same_parent_actor_adam_moments_random_states_and_pool_are_restored(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                trainer = self.trainer(arm)
                self.assertEqual(state_hash(trainer.policy.state_dict()), self.saved["state_sha256"])
                self.assertEqual(state_hash(trainer.reference.state_dict()), self.saved["state_sha256"])
                self.assertEqual(nested_hash(trainer.optimizer.state_dict()["state"]),
                                 nested_hash(self.saved["optimizer_state"]["state"]))
                self.assertEqual(nested_hash(trainer.checkpoint()["rng_states"]), nested_hash(self.saved["rng_states"]))
                self.assertEqual(trainer.pool.by_sequence, self.saved["elite_pool"])
                self.assertTrue(all(not p.requires_grad for p in trainer.reference.parameters()))
                self.assertEqual(trainer.settings["learning_rate"], 3e-4 if arm in ("patternboost", "grpo") else 3e-5)

    def test_current_grpo_update_is_identical_to_allocation_reference(self):
        reference = AllocationTrainer(self.data, 7, 1, config=self.config)
        reference.restore(self.saved)
        reference.reinforcement_step(time.monotonic() + 60)
        reference_hash = state_hash(reference.policy.state_dict())
        reference_optimizer_hash = nested_hash(reference.optimizer.state_dict())
        trainer = self.trainer("grpo")
        trainer.algorithm_step(time.monotonic() + 60)
        self.assertEqual(state_hash(trainer.policy.state_dict()), reference_hash)
        self.assertEqual(nested_hash(trainer.optimizer.state_dict()), reference_optimizer_hash)
        self.assertEqual(trainer.counters["rl_updates"], reference.counters["rl_updates"])
        self.assertEqual(trainer.counters["generated_folds"], reference.counters["generated_folds"])

    def test_new_algorithms_change_actor_and_leave_reference_frozen(self):
        for arm in ("dr_grpo", "reinforce", "dpo"):
            with self.subTest(arm=arm):
                trainer = self.trainer(arm)
                trainer.sample = lambda *_args, **_kwargs: [copy.deepcopy(row) for row in [self.low, self.high] * 4]
                updated = trainer.algorithm_step(time.monotonic() + 60)
                self.assertTrue(updated)
                self.assertNotEqual(state_hash(trainer.policy.state_dict()), self.saved["state_sha256"])
                self.assertEqual(state_hash(trainer.reference.state_dict()), self.saved["state_sha256"])
                self.assertTrue(all(p.grad is None for p in trainer.reference.parameters()))
                if arm == "dpo":
                    self.assertEqual(trainer.counters["preference_pairs"], 4)
                    self.assertEqual(trainer.counters["preference_updates"], 1)

    def test_interrupted_groups_do_not_update_any_algorithm(self):
        for arm in ("grpo", "dr_grpo", "reinforce", "dpo"):
            with self.subTest(arm=arm):
                trainer = self.trainer(arm)
                trainer.sample = lambda *_args, **_kwargs: [{**self.high, "interrupted": True}] * 8
                self.assertFalse(trainer.algorithm_step(time.monotonic() + 60))
                self.assertEqual(state_hash(trainer.policy.state_dict()), self.saved["state_sha256"])
                self.assertEqual(trainer.counters["interrupted_folds"] - trainer.initial_counters["interrupted_folds"], 8)

    def test_reference_rollout_is_deterministic_rng_free_and_deadline_aware(self):
        trainer = self.trainer("reinforce")
        before = nested_hash(trainer.checkpoint()["rng_states"])
        first = greedy_policy_rollout(trainer.reference, "HHHHHHHH", time.monotonic() + 60)
        second = greedy_policy_rollout(trainer.reference, "HHHHHHHH", time.monotonic() + 60)
        self.assertEqual(first, second)
        self.assertEqual(before, nested_hash(trainer.checkpoint()["rng_states"]))
        expired = greedy_policy_rollout(trainer.reference, "HHHHHHHH", time.monotonic() - 1)
        self.assertTrue(expired["interrupted"])
        self.assertFalse(expired["complete"])

    def test_short_runs_record_lineage_final_endpoints_and_independent_counters(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                destination = self.folder / arm
                result = run(self.data, self.path, arm, 7, 0.6, 1, destination)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["training_seconds"], 7200.6)
                self.assertEqual(result["additional_training_seconds"], 0.6)
                self.assertEqual(result["base_training_seconds"], 7200)
                self.assertEqual(result["parent_file_sha256"], file_sha256(self.path))
                self.assertEqual(result["initial_state_sha256"], self.saved["state_sha256"])
                self.assertEqual(result["initial_timing"], self.saved["timing"])
                for phase, cumulative in result["timing"].items():
                    self.assertAlmostEqual(result["segment_timing"][phase],
                                           cumulative - self.saved["timing"][phase])
                self.assertIn("cumulative", result["timing_note"])
                self.assertGreaterEqual(result["compute_elapsed_s"], 0.6)
                self.assertGreater(result["checkpoint_io_s"], 0)
                saved = self.torch.load(destination / "final.pt", weights_only=False)
                validate_finetune_checkpoint(saved, result["dataset_sha256"], 7, arm, file_sha256(self.path))
                self.assertFalse(saved["selection"]["selected"])
                self.assertFalse(saved["test_evaluated"])
                if arm != "patternboost":
                    self.assertEqual(result["segment_counters"]["supervised_updates"], 0)
                self.assertEqual(result["checkpoint_records"][0]["file_sha256"], file_sha256(destination / "final.pt"))
        with self.assertRaisesRegex(ValueError, "fresh empty"):
            run(self.data, self.path, "grpo", 7, 1, 1, self.folder / "grpo")

    def test_validator_rejects_tampering_and_validation_selection(self):
        trainer = self.trainer("dpo")
        saved = trainer.checkpoint()
        validate_finetune_checkpoint(saved, trainer.dataset_sha256, 7, "dpo", trainer.parent_file_sha256)
        for field, value in (("dataset_sha256", "wrong"), ("parent_state_sha256", "wrong"),
                             ("training_seconds", 999), ("final_test_enabled", True),
                             ("selection", {"selected": True}), ("additional_training_seconds", float("nan")),
                             ("mode", "patternboost"), ("stage", "grpo")):
            altered = {**saved, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_finetune_checkpoint(altered, trainer.dataset_sha256, 7, "dpo", trainer.parent_file_sha256)
        altered = copy.deepcopy(saved)
        next(iter(altered["reference_state"].values())).view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_finetune_checkpoint(altered, trainer.dataset_sha256)

    def test_no_updates_and_signal_stop_cannot_be_complete(self):
        def identical(*_args, **_kwargs):
            return [copy.deepcopy(self.high)] * 8
        with mock.patch(f"{self.parent.sample.__module__}.sample", side_effect=identical):
            result = run(self.data, self.path, "dr_grpo", 7, 0.1, 1, self.folder / "identical")
        self.assertEqual(result["status"], "insufficient_training")
        def stop(trainer, _deadline):
            trainer.stopped = True
            return False
        with mock.patch.object(FineTuneTrainer, "algorithm_step", stop):
            result = run(self.data, self.path, "reinforce", 7, 3, 1, self.folder / "interrupted")
        self.assertEqual(result["status"], "interrupted")
        self.assertLess(result["additional_training_seconds"], 3)

    def test_diagnostics_record_batch_context_without_global_diversity_claim(self):
        trainer = self.trainer("grpo")
        trainer._last_rows = [self.low, self.high, copy.deepcopy(self.high)]
        trainer.diagnose(time.monotonic() + 60)
        self.assertEqual(trainer.diagnostics["sampled_sequence_count"], 1)
        self.assertEqual(trainer.diagnostics["complete_sequence_count"], 1)
        self.assertAlmostEqual(trainer.diagnostics["unique_complete_fold_fraction"], 2 / 3)
        self.assertIn("not global diversity", trainer.diagnostics["diversity_note"])
        self.assertIn("not directly comparable", trainer.diagnostics["diversity_note"])


if __name__ == "__main__":
    unittest.main()
