"""Final-study adapters preserve objectives, provenance and split isolation."""
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
    from .finetune_train import FineTuneTrainer
    from .final_train import (FinalParentTrainer, FinalTrainer, BRANCH_ARMS, SCRATCH_ARMS,
                             MODEL, train_parent, train_endpoint, validate_final_checkpoint)
    from .train import state_hash
except ImportError:
    import geometry
    from allocation_train import AllocationTrainer, file_sha256, nested_hash
    from finetune_train import FineTuneTrainer
    from final_train import (FinalParentTrainer, FinalTrainer, BRANCH_ARMS, SCRATCH_ARMS,
                            MODEL, train_parent, train_endpoint, validate_final_checkpoint)
    from train import state_hash


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class FinalTrainingTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory(prefix="hp-final-training-")
        self.root = Path(self.temp.name)
        self.data = self.root / "data.json"
        self.data.write_text(json.dumps({"train": [{"sequence": "HHHHHHHH"}],
                                        "validation": [{"sequence": "HHHPHHHH"}],
                                        "test": [{"sequence": "HHPHPHHH"}]}))
        self.config = {"model": MODEL, "allocation": {"initial_data_seconds": 0.05}}
        self.parent = FinalParentTrainer(self.data, self.config, 1, 1)
        self.parent.initialize_examples(time.monotonic() + 0.15)
        self.parent.supervised_step()
        self.parent.training_seconds = self.parent.segment_training_seconds = 7200.0
        self.parent.measured_training_seconds = 7200.0
        self.parent.final_status = "complete"
        self.saved = self.parent.checkpoint()
        self.path = self.root / "parent.pt"
        torch.save(self.saved, self.path)
        low = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
               (4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 3, 0)]
        high = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)]
        self.rows = [geometry.trajectory("HHHHHHHH", points) for points in (low, high)] * 4

    def tearDown(self):
        self.temp.cleanup()

    def branch(self, arm):
        return FinalTrainer(self.data, self.config, 1, 1, arm, self.path)

    def test_branch_restores_actor_adam_rng_and_elite_without_validation_access(self):
        for arm in BRANCH_ARMS:
            with self.subTest(arm=arm):
                trainer = self.branch(arm)
                state = trainer.checkpoint()
                self.assertEqual(set(trainer.data), {"train"})
                self.assertEqual(state_hash(trainer.policy.state_dict()), self.saved["state_sha256"])
                self.assertEqual(state_hash(trainer.reference.state_dict()), self.saved["state_sha256"])
                self.assertEqual(nested_hash(trainer.optimizer.state_dict()["state"]),
                                 nested_hash(self.saved["optimizer_state"]["state"]))
                self.assertEqual(nested_hash(state["rng_states"]), nested_hash(self.saved["rng_states"]))
                self.assertEqual(trainer.pool.by_sequence, self.saved["elite_pool"])
                self.assertEqual(trainer.settings["learning_rate"], 3e-5)
                self.assertEqual(state["final_campaign"]["reference_source"], "frozen_patternboost_parent")
                self.assertEqual(state["parent_file_sha256"], file_sha256(self.path))

    def test_scratch_uses_same_random_actor_and_no_supervised_examples(self):
        expected_initial = self.saved["initial_state_sha256"]
        for arm in SCRATCH_ARMS:
            with self.subTest(arm=arm):
                trainer = FinalTrainer(self.data, self.config, 1, 1, arm)
                self.assertEqual(set(trainer.data), {"train"})
                self.assertEqual(state_hash(trainer.policy.state_dict()), expected_initial)
                self.assertEqual(state_hash(trainer.reference.state_dict()), expected_initial)
                self.assertFalse(trainer.optimizer.state_dict()["state"])
                self.assertFalse(trainer.pool.by_sequence)
                self.assertEqual(trainer.base_training_seconds, 0)
                self.assertEqual(trainer.counters["supervised_updates"], 0)
                self.assertEqual(trainer.settings["learning_rate"], 3e-5)
                self.assertEqual(trainer.checkpoint()["final_campaign"]["reference_source"],
                                 "frozen_random_initialization")
                trainer.sample = lambda *_args, **_kwargs: copy.deepcopy(self.rows)
                self.assertTrue(trainer.algorithm_step(time.monotonic() + 60))
                self.assertNotEqual(state_hash(trainer.policy.state_dict()), expected_initial)
                self.assertEqual(state_hash(trainer.reference.state_dict()), expected_initial)
                self.assertTrue(all(p.grad is None for p in trainer.reference.parameters()))
                self.assertEqual(trainer.counters["supervised_updates"], 0)

    def test_branch_update_is_identical_to_screened_objective(self):
        legacy = copy.deepcopy(self.saved)
        legacy["kind"] = "allocation_training_state"
        path = self.root / "legacy.pt"
        self.torch.save(legacy, path)
        for arm in ("grpo", "dr_grpo", "reinforce", "dpo"):
            with self.subTest(arm=arm):
                expected = FineTuneTrainer(self.data, path, arm, 1, 1, learning_rate=3e-5)
                expected.sample = lambda *_args, **_kwargs: copy.deepcopy(self.rows)
                expected.algorithm_step(time.monotonic() + 60)
                actual = self.branch(arm)
                actual.sample = lambda *_args, **_kwargs: copy.deepcopy(self.rows)
                actual.algorithm_step(time.monotonic() + 60)
                self.assertEqual(state_hash(actual.policy.state_dict()), state_hash(expected.policy.state_dict()))
                self.assertEqual(nested_hash(actual.optimizer.state_dict()), nested_hash(expected.optimizer.state_dict()))
                self.assertEqual(actual.counters, expected.counters)

    def test_short_runs_save_complete_fixed_endpoints_and_lineage(self):
        parent = train_parent(self.data, self.config, self.root / "new_parent", 1, 0.7, 1)
        self.assertEqual(parent["status"], "complete")
        self.assertEqual(parent["training_seconds"], 0.7)
        self.assertEqual(parent["checkpoint_sha256"], file_sha256(parent["checkpoint_path"]))
        cases = [("grpo", self.path), ("scratch_grpo", None), ("scratch_dr_grpo", None),
                 ("scratch_reinforce", None)]
        for arm, checkpoint in cases:
            with self.subTest(arm=arm):
                result = train_endpoint(self.data, self.config, self.root / arm, 1, arm, 0.5, 1, checkpoint)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["training_seconds"], 7200.5 if checkpoint else 0.5)
                self.assertEqual(result["segment_training_seconds"], 0.5)
                self.assertGreaterEqual(result["compute_elapsed_s"], 0.5)
                self.assertGreater(result["checkpoint_io_s"], 0)
                saved = self.torch.load(result["checkpoint_path"], weights_only=False)
                validate_final_checkpoint(saved, result["dataset_sha256"], seed=1, arm=arm)
                self.assertEqual(result["checkpoint_sha256"], file_sha256(result["checkpoint_path"]))
                self.assertFalse(saved["selection"]["selected"])
                self.assertFalse(saved["test_evaluated"])
                self.assertEqual(result["segment_counters"]["supervised_updates"], 0)

    def test_rejects_old_selected_wrong_seed_and_architecture_parents(self):
        cases = [("kind", "allocation_training_state"), ("status", "interrupted"),
                 ("selection", {"selected": True}), ("seed", 2),
                 ("model_config", {**MODEL, "width": 128}), ("test_evaluated", True)]
        for field, value in cases:
            with self.subTest(field=field):
                altered = copy.deepcopy(self.saved)
                altered[field] = value
                path = self.root / "invalid.pt"
                self.torch.save(altered, path)
                with self.assertRaises(ValueError):
                    FinalTrainer(self.data, self.config, 1, 1, "grpo", path)
        with self.assertRaisesRegex(ValueError, "frozen"):
            FinalParentTrainer(self.data, {"model": {"width": 128}}, 1, 1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            FinalParentTrainer(self.data, {"training": {"group_size": 16}}, 1, 1)
        for arm, path in (("grpo", None), ("scratch_grpo", self.path)):
            with self.assertRaisesRegex(ValueError, "require a parent"):
                FinalTrainer(self.data, self.config, 1, 1, arm, path)

    def test_bad_checkpoint_hashes_and_split_flags_are_rejected(self):
        validate_final_checkpoint(self.saved, self.saved["dataset_sha256"], 1, "parent")
        altered = copy.deepcopy(self.saved)
        next(iter(altered["model_state"].values())).view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_final_checkpoint(altered, self.saved["dataset_sha256"])
        for field, value in (("training_data_split", "validation"), ("segment_training_seconds", 0),
                             ("parent_file_sha256", "old-pilot"), ("training_seconds", float("nan"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_final_checkpoint({**self.saved, field: value}, self.saved["dataset_sha256"])

    def test_interruption_and_no_updates_cannot_be_called_complete(self):
        def stop(trainer, _deadline):
            trainer.stopped = True
            return False
        with mock.patch.object(FinalTrainer, "algorithm_step", stop):
            result = train_endpoint(self.data, self.config, self.root / "stopped", 1, "scratch_grpo", 2, 1)
        self.assertEqual(result["status"], "interrupted")
        self.assertLess(result["segment_training_seconds"], 2)
        saved = self.torch.load(result["checkpoint_path"], weights_only=False)
        with self.assertRaisesRegex(ValueError, "not complete"):
            validate_final_checkpoint(saved, result["dataset_sha256"])
        with mock.patch.object(FinalTrainer, "algorithm_step", return_value=False):
            result = train_endpoint(self.data, self.config, self.root / "zero", 1, "scratch_grpo", 0.05, 1)
        self.assertEqual(result["status"], "insufficient_training")
        with self.assertRaisesRegex(ValueError, "fresh empty"):
            train_endpoint(self.data, self.config, self.root / "zero", 1, "scratch_grpo", 0.05, 1)


if __name__ == "__main__":
    unittest.main()
