"""Exact resume and short disposable allocation checks; no final test inference."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

try:
    from .allocation_train import AllocationTrainer, file_sha256, nested_hash, run
    from .train import state_hash
except ImportError:
    from allocation_train import AllocationTrainer, file_sha256, nested_hash, run
    from train import state_hash


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required")
class AllocationTrainingTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory(prefix="hp-allocation-state-")
        self.folder = Path(self.temporary.name)
        self.data = self.folder / "data.json"
        self.data.write_text(json.dumps({
            "train": [{"sequence": "HHHHHHHH"}, {"sequence": "HHPPHHHH"}],
            "validation": [{"sequence": "HHHPHHHH"}],
            "test": [{"sequence": "HHPHPHHH"}]}))
        self.config = {"model": {"width": 16, "heads": 2, "layers": 1, "dropout": 0.0},
                       "training": {"batch_size": 4, "group_size": 8, "elite_size": 2},
                       "allocation": {"initial_data_seconds": 0.1,
                                      "milestones_seconds": [1.6, 2.4, 3.2]}}

    def tearDown(self):
        self.temporary.cleanup()

    def trainer(self, seed=7, config=None, data=None):
        return AllocationTrainer(data or self.data, seed, 1, config or self.config)

    def populated(self):
        trainer = self.trainer()
        trainer.initialize_examples(time.monotonic() + 0.15)
        trainer.supervised_step()
        return trainer

    def continuation(self, trainer):
        trainer.supervised_step()
        trainer.refresh(time.monotonic() + 60)
        trainer.supervised_step()
        trainer.reinforcement_step(time.monotonic() + 60)
        return {"model": state_hash(trainer.policy.state_dict()),
                "optimizer": nested_hash(trainer.optimizer.state_dict()),
                "pool": copy.deepcopy(trainer.pool.by_sequence),
                "counters": dict(trainer.counters),
                "global_random": random.random(), "local_random": trainer.rng.random(),
                "global_torch": self.torch.rand(3).tolist(),
                "local_torch": self.torch.rand(3, generator=trainer.generator).tolist()}

    def test_saved_state_matches_uninterrupted_continuation(self):
        trainer = self.populated()
        checkpoint = trainer.checkpoint()
        path = self.folder / "state.pt"
        self.torch.save(checkpoint, path)
        uninterrupted = self.continuation(trainer)
        restored = self.trainer()
        restored.restore(self.torch.load(path, map_location="cpu", weights_only=False))
        self.assertEqual(restored.pool.by_sequence, checkpoint["elite_pool"])
        self.assertEqual(nested_hash(restored.optimizer.state_dict()), checkpoint["optimizer_sha256"])
        continued = self.continuation(restored)
        self.assertEqual(uninterrupted, continued)
        self.assertGreater(continued["counters"]["rl_updates"], 0)

    def test_resume_rejects_seed_data_config_and_hash_mismatch(self):
        checkpoint = self.populated().checkpoint()
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.trainer(seed=8).restore(checkpoint)
        changed = copy.deepcopy(self.config)
        changed["training"]["learning_rate"] = 0.002
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            self.trainer(config=changed).restore(checkpoint)
        other_data = self.folder / "other.json"
        other_data.write_text(self.data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "dataset_sha256 mismatch"):
            self.trainer(data=other_data).restore(checkpoint)
        altered = copy.deepcopy(checkpoint)
        first = next(iter(altered["model_state"]))
        altered["model_state"][first].view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            self.trainer().restore(altered)

    def test_branch_handoff_retains_optimizer_rng_and_pool(self):
        trainer = self.populated()
        trainer.training_seconds = 1.6
        checkpoint = trainer.checkpoint()
        restored = self.trainer()
        restored.restore(checkpoint)
        before_rng = nested_hash(restored.checkpoint()["rng_states"])
        restored.begin_rl_branch(self.folder / "state.pt", "parent-file-hash")
        self.assertEqual(restored.handoff["pretraining_sha256"], restored.handoff["rl_initial_sha256"])
        self.assertEqual(restored.handoff["pretraining_optimizer_sha256"], restored.handoff["rl_initial_optimizer_sha256"])
        self.assertEqual(before_rng, nested_hash(restored.checkpoint()["rng_states"]))
        self.assertEqual(checkpoint["elite_pool"], restored.pool.by_sequence)
        self.assertEqual(restored.mode, "rl_branch")
        with self.assertRaisesRegex(ValueError, "PatternBoost stage"):
            restored.begin_rl_branch(self.folder / "state.pt", "parent-file-hash")

    def test_short_parent_branch_and_scratch_complete(self):
        parent = run(self.data, "patternboost", 7, 3.2, 1, self.folder / "parent", self.config)
        self.assertEqual(parent["status"], "complete")
        self.assertGreater(parent["counters"]["supervised_updates"], 0)
        self.assertEqual([record["filename"] for record in parent["checkpoint_records"]],
                         ["stage_1.6.pt", "stage_2.4.pt", "stage_3.2.pt", "final.pt"])
        checkpoint = self.folder / "parent" / "stage_1.6.pt"
        branch = run(self.data, "rl_branch", 7, 1.6, 1, self.folder / "branch", self.config, checkpoint)
        self.assertEqual(branch["status"], "complete")
        self.assertGreater(branch["segment_counters"]["rl_updates"], 0)
        self.assertEqual(branch["segment_counters"]["supervised_updates"], 0)
        self.assertEqual(branch["segment_counters"]["supervised_examples"], 0)
        self.assertEqual(branch["lineage"]["parent_file_sha256"], file_sha256(checkpoint))
        self.assertEqual(branch["initial_state_sha256"], parent["checkpoint_records"][0]["state_sha256"])
        self.assertEqual(branch["training_seconds"], 3.2)
        self.assertEqual(branch["handoff"]["supervised_updates"],
                         self.torch.load(checkpoint, weights_only=False)["counters"]["supervised_updates"])
        self.assertFalse(branch["test_evaluated"])
        later_checkpoint = self.folder / "parent" / "stage_2.4.pt"
        later = run(self.data, "rl_branch", 7, 0.8, 1, self.folder / "later_branch",
                    self.config, later_checkpoint)
        self.assertEqual(later["status"], "complete")
        self.assertGreater(later["segment_counters"]["rl_updates"], 0)
        self.assertEqual(later["segment_counters"]["supervised_updates"], 0)
        self.assertEqual(later["lineage"]["pretraining_seconds"], 2.4)
        self.assertEqual(later["training_seconds"], 3.2)
        self.assertEqual(later["initial_state_sha256"], parent["checkpoint_records"][1]["state_sha256"])
        self.assertEqual(later["lineage"]["parent_file_sha256"], file_sha256(later_checkpoint))
        for result, folder in [(parent, "parent"), (branch, "branch"), (later, "later_branch")]:
            for record in result["checkpoint_records"]:
                self.assertEqual(record["file_sha256"], file_sha256(self.folder / folder / record["filename"]))
            self.assertGreaterEqual(result["elapsed_s"], result["compute_elapsed_s"])
        scratch = run(self.data, "rl_scratch", 7, 1.2, 1, self.folder / "scratch", self.config)
        self.assertEqual(scratch["status"], "complete")
        self.assertEqual(scratch["counters"]["supervised_updates"], 0)
        self.assertGreater(scratch["counters"]["rl_updates"], 0)
        with self.assertRaisesRegex(ValueError, "fresh empty"):
            run(self.data, "rl_scratch", 7, 1, 1, self.folder / "scratch", self.config)
        with self.assertRaisesRegex(ValueError, "declared allocation"):
            run(self.data, "rl_branch", 7, 1, 1, self.folder / "bad_branch", self.config, checkpoint)


if __name__ == "__main__":
    unittest.main()
