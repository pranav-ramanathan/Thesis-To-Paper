"""Frozen final-study training adapters; no validation or test-fold evaluation.

Objective implementations are inherited from the screened trainers. All branches
start from a fresh final-study PatternBoost parent, while scratch policies start
from the same seed's random initialization. REINFORCE always uses a frozen policy
argmax baseline: the PB parent for a branch, the initial random actor for scratch.
These are fixed configurations, not an equally tuned algorithm competition.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import signal
import time

try:
    from .allocation_train import AllocationTrainer, file_sha256, nested_hash
    from .finetune_train import FineTuneTrainer, DIAGNOSTIC_EVERY, DPO_BETA
    from .train import DEFAULTS, atomic_json, state_hash
except ImportError:
    from allocation_train import AllocationTrainer, file_sha256, nested_hash
    from finetune_train import FineTuneTrainer, DIAGNOSTIC_EVERY, DPO_BETA
    from train import DEFAULTS, atomic_json, state_hash

MODEL = {"width": 64, "heads": 4, "layers": 2, "dropout": 0.0}
BRANCH_ARMS = ("patternboost", "grpo", "dr_grpo", "reinforce", "dpo")
SCRATCH_ARMS = ("scratch_grpo", "scratch_dr_grpo", "scratch_reinforce")
ENDPOINT_ARMS = BRANCH_ARMS + SCRATCH_ARMS
PARENT_LR = 3e-4
ENDPOINT_LR = 3e-5


def _config(config, learning_rate):
    """Only the declared trainer settings enter saved training state."""
    config = copy.deepcopy(config)
    model = {**MODEL, **config.get("model", {})}
    if model != MODEL:
        raise ValueError("Final training requires the frozen 64/4/2 FP32 architecture")
    settings = {**DEFAULTS, **config.get("training", {})}
    for name in ("batch_size", "group_size", "elite_size", "trap_penalty", "grpo_clip",
                 "grpo_epochs", "grpo_kl_beta", "temperature", "gradient_clip",
                 "refresh_every_updates", "regrow_probability", "greedy_completion_probability"):
        if settings[name] != DEFAULTS[name]:
            raise ValueError(f"Final training setting is frozen: {name}")
    settings["learning_rate"] = learning_rate
    allocation = {"initial_data_seconds": 900, **config.get("allocation", {})}
    return {"model": model, "training": settings, "allocation": allocation}


def _identity(trainer, role, arm, source_parent, initial_hash, reference_source):
    return {
        "kind": "final_policy_training_state", "role": role, "arm": arm,
        "status": trainer.final_status,
        "segment_training_seconds": trainer.segment_training_seconds,
        "parent_file_sha256": trainer.parent_file_sha256 if source_parent else None,
        "parent_state_sha256": initial_hash if source_parent else None,
        "initial_state_sha256": initial_hash, "training_data_split": "train",
        "final_campaign": {"schema": 1, "role": role, "arm": arm,
                           "training_data_split": "train", "fresh_initialization": not bool(source_parent),
                           "reference_source": reference_source},
    }


class FinalParentTrainer(AllocationTrainer):
    def __init__(self, data, config, seed, threads, emit=None):
        super().__init__(data, seed, threads, _config(config, PARENT_LR), emit)
        # Split checks above inspect sequence membership; optimization can access
        # only the training rows after that validation.
        self.data = {"train": self.data["train"]}
        self.initial_state_sha256 = state_hash(self.policy.state_dict())
        self.final_status = "running"
        self.segment_training_seconds = 0.0

    def checkpoint(self):
        saved = super().checkpoint()
        saved["origin_kind"] = saved["kind"]
        saved.update(_identity(self, "parent", "parent", None, self.initial_state_sha256, None))
        return saved


class FinalTrainer(FineTuneTrainer):
    """Reuse screened objectives without pretending scratch has a PB parent."""
    def __init__(self, data, config, seed, threads, arm, parent_checkpoint=None, emit=None):
        if arm not in ENDPOINT_ARMS:
            raise ValueError("Unknown final training arm")
        scratch = arm in SCRATCH_ARMS
        if scratch == (parent_checkpoint is not None):
            raise ValueError("Only PatternBoost branches require a parent checkpoint")
        setup = _config(config, ENDPOINT_LR if scratch else PARENT_LR)
        AllocationTrainer.__init__(self, data, seed, threads, setup, emit)
        parent = None
        if parent_checkpoint:
            parent = self.torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
            validate_final_checkpoint(parent, self.dataset_sha256, seed=seed, arm="parent")
            legacy = copy.deepcopy(parent)
            legacy["kind"] = "allocation_training_state"
            self.restore(legacy)
            if not self.data_initialized or self.counters["supervised_updates"] < 1:
                raise ValueError("Final branches require a trained PatternBoost parent")
        self.data = {"train": self.data["train"]}
        self.final_arm = arm
        self.arm = arm.removeprefix("scratch_")
        self.source_parent = Path(parent_checkpoint) if parent_checkpoint else None
        self.final_status = "running"
        self.segment_training_seconds = 0.0
        self.base_training_seconds = self.training_seconds
        self.base_measured_seconds = self.measured_training_seconds
        self.parent_file_sha256 = file_sha256(parent_checkpoint) if parent_checkpoint else None
        self.parent_state_sha256 = state_hash(self.policy.state_dict())
        self.parent_optimizer_sha256 = nested_hash(self.optimizer.state_dict())
        self.initial_timing = dict(self.timing)
        self.reference = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.max_moves = max(len(row["sequence"]) - 2 for row in self.data["train"])
        for group in self.optimizer.param_groups:
            group["lr"] = ENDPOINT_LR
        self.settings["learning_rate"] = ENDPOINT_LR
        self.condition, self.mode, self.stage = self.arm, "finetune", self.arm
        self.additional_training_seconds = 0.0
        self.counters.update({"dpo_updates": 0, "preference_updates": 0, "preference_pairs": 0,
                              "no_preference_groups": 0, "reference_rollouts": 0,
                              "reference_complete_folds": 0, "reference_interrupted_folds": 0})
        self.initial_counters = dict(self.counters)
        self.handoff = ({"pretraining_sha256": self.parent_state_sha256,
                         "finetune_initial_sha256": self.parent_state_sha256,
                         "pretraining_optimizer_sha256": self.parent_optimizer_sha256,
                         "initial_optimizer_sha256_after_lr_override": nested_hash(self.optimizer.state_dict()),
                         "optimizer_moments_retained": True, "rng_state_retained": True,
                         "elite_pool_retained": True, "learning_rate_override": ENDPOINT_LR}
                        if parent is not None else None)
        self.lineage = {"root_seed": seed, "parent_file_sha256": self.parent_file_sha256,
                        "parent_state_sha256": self.parent_state_sha256 if parent is not None else None,
                        "pretraining_seconds": self.base_training_seconds,
                        "initialization": "fresh_random" if scratch else "fresh_final_patternboost_parent"}
        self.diagnostics = {"records": 0, "sampled_logp_drift_mean": None,
                            "unique_complete_fold_fraction": None,
                            "sampled_sequence_count": None, "complete_sequence_count": None,
                            "gradient_norm_after_clip": None, "reward_sd": None,
                            "reward_min": None, "reward_max": None,
                            "interpretation": "Sampled current-minus-frozen-reference action log probability is not exact KL.",
                            "diversity_note": "Within-batch uniqueness; PB can mix input sequences while policy-gradient groups share one sequence."}
        self._last_rows = []
        sampler = self.sample
        def capture(*args, **kwargs):
            rows = sampler(*args, **kwargs)
            self._last_rows = rows
            return rows
        self.sample = capture

    def checkpoint(self):
        saved = super().checkpoint()
        saved["origin_kind"] = saved["kind"]
        source = "frozen_patternboost_parent" if self.source_parent else "frozen_random_initialization"
        saved.update(_identity(self, "endpoint", self.final_arm, self.source_parent,
                               self.parent_state_sha256, source))
        return saved


def validate_final_checkpoint(saved, data_hash, seed=None, arm=None, parent_sha=None, require_complete=True):
    """Validate fixed-endpoint identity, tensor hashes, training lineage and split."""
    if saved.get("schema") != 1 or saved.get("kind") != "final_policy_training_state":
        raise ValueError("Expected a final-study training state")
    if saved.get("dataset_sha256") != data_hash:
        raise ValueError("Final training dataset mismatch")
    role, actual_arm = saved.get("role"), saved.get("arm")
    if role not in ("parent", "endpoint") or (role == "parent") != (actual_arm == "parent"):
        raise ValueError("Final training role/arm mismatch")
    if actual_arm != "parent" and actual_arm not in ENDPOINT_ARMS:
        raise ValueError("Unknown final checkpoint arm")
    for field, expected in (("seed", seed), ("arm", arm), ("parent_file_sha256", parent_sha)):
        if expected is not None and saved.get(field) != expected:
            raise ValueError(f"Final training {field} mismatch")
    if require_complete and saved.get("status") != "complete":
        raise ValueError("Final training is not complete")
    if saved.get("model_config") != MODEL:
        raise ValueError("Final model architecture mismatch")
    if (saved.get("training_data_split") != "train" or saved.get("test_evaluated") is not False
            or saved.get("final_test_enabled") is not False or saved.get("selection", {}).get("selected") is not False):
        raise ValueError("Final training must use only train with fixed endpoints")
    meta = saved.get("final_campaign", {})
    if (meta.get("schema") != 1 or meta.get("role") != role or meta.get("arm") != actual_arm
            or meta.get("training_data_split") != "train"):
        raise ValueError("Final campaign metadata mismatch")
    for field, hash_field in (("model_state", "state_sha256"), ("optimizer_state", "optimizer_sha256")):
        hasher = state_hash if field == "model_state" else nested_hash
        if field not in saved or hasher(saved[field]) != saved.get(hash_field):
            raise ValueError(f"Final training {field} hash mismatch")
    if any(value.is_floating_point() and str(value.dtype) != "torch.float32"
           for value in saved["model_state"].values()):
        raise ValueError("Final actor must use FP32")
    for field in ("training_seconds", "segment_training_seconds", "measured_training_seconds"):
        value = saved.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid final training duration: {field}")
    if not all(key in saved for key in ("rng_states", "elite_pool", "counters", "initial_state_sha256")):
        raise ValueError("Incomplete final training state")
    settings = saved.get("effective_training_settings", {})
    expected_lr = PARENT_LR if role == "parent" else ENDPOINT_LR
    if settings.get("learning_rate") != expected_lr:
        raise ValueError("Final learning rate mismatch")
    normalized = _config(saved.get("config", {}), expected_lr)
    for name in ("batch_size", "group_size", "elite_size", "trap_penalty", "grpo_clip",
                 "grpo_epochs", "grpo_kl_beta", "temperature", "gradient_clip",
                 "refresh_every_updates", "regrow_probability", "greedy_completion_probability"):
        if settings.get(name) != normalized["training"][name]:
            raise ValueError(f"Final objective setting mismatch: {name}")
    if any(group.get("lr") != expected_lr for group in saved["optimizer_state"].get("param_groups", [])):
        raise ValueError("Final optimizer learning rate mismatch")
    expected_origin = "allocation_training_state" if role == "parent" else "finetune_training_state"
    if saved.get("origin_kind") != expected_origin:
        raise ValueError("Final checkpoint origin mismatch")
    fresh = role == "parent" or actual_arm in SCRATCH_ARMS
    if meta.get("fresh_initialization") is not fresh:
        raise ValueError("Final initialization provenance mismatch")
    if fresh and (saved.get("parent_file_sha256") is not None or saved.get("parent_state_sha256") is not None):
        raise ValueError("Fresh training cannot have a parent")
    base = 0.0 if role == "parent" else saved.get("base_training_seconds")
    if not isinstance(base, (int, float)) or not math.isfinite(base) or base < 0:
        raise ValueError("Invalid parent training duration")
    if abs(saved["training_seconds"] - base - saved["segment_training_seconds"]) > 1e-6:
        raise ValueError("Final training lineage duration mismatch")
    if role == "parent":
        if any(saved.get(field) != "patternboost" for field in ("condition", "mode", "stage")):
            raise ValueError("Final parent mode mismatch")
        if meta.get("reference_source") is not None:
            raise ValueError("Parent does not have a frozen reference")
        if require_complete and (saved["counters"].get("supervised_updates", 0) < 1 or not saved.get("data_initialized")):
            raise ValueError("Parent has no supervised training")
    else:
        objective = actual_arm.removeprefix("scratch_")
        if saved.get("condition") != objective or saved.get("stage") != objective or saved.get("mode") != "finetune":
            raise ValueError("Final endpoint objective mismatch")
        if saved.get("additional_training_seconds") != saved["segment_training_seconds"]:
            raise ValueError("Final endpoint segment duration mismatch")
        if require_complete and sum(saved.get("segment_counters", {}).get(key, 0)
                                    for key in ("supervised_updates", "rl_updates", "dpo_updates")) < 1:
            raise ValueError("Final endpoint has no training updates")
        if state_hash(saved.get("reference_state", {})) != saved.get("reference_state_sha256"):
            raise ValueError("Final reference hash mismatch")
        if saved["reference_state_sha256"] != saved["initial_state_sha256"]:
            raise ValueError("Final reference must remain frozen at initialization")
        expected_source = "frozen_random_initialization" if fresh else "frozen_patternboost_parent"
        if meta.get("reference_source") != expected_source:
            raise ValueError("Final reference provenance mismatch")
        if fresh:
            if base != 0 or saved["counters"].get("supervised_updates") != 0 or saved["elite_pool"]:
                raise ValueError("Scratch policy contains supervised pretraining")
        elif (base <= 0 or not saved.get("parent_file_sha256")
              or saved.get("parent_state_sha256") != saved["initial_state_sha256"]
              or saved.get("handoff", {}).get("finetune_initial_sha256") != saved["initial_state_sha256"]):
            raise ValueError("Final parent handoff mismatch")
    return saved


def _run(data, config, output, seed, seconds, threads, arm, parent_checkpoint=None):
    started = time.monotonic()
    if (not math.isfinite(seconds) or seconds <= 0 or not isinstance(seed, int) or seed < 0
            or not isinstance(threads, int) or threads < 1):
        raise ValueError("Invalid final training request")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Final training output must be a fresh empty directory")
    handlers, trainer = {}, None
    handle = (output / "events.jsonl").open("w", buffering=1)
    def emit(kind, **fields):
        handle.write(json.dumps({"type": kind, "elapsed_s": time.monotonic() - started,
                                 "arm": arm, **fields}, sort_keys=True) + "\n")
    try:
        trainer = (FinalParentTrainer(data, config, seed, threads, emit) if arm == "parent" else
                   FinalTrainer(data, config, seed, threads, arm, parent_checkpoint, emit))
        trainer.timing["initialization_s"] += time.monotonic() - started
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda *_: setattr(trainer, "stopped", True))
        initial_counters = dict(trainer.counters)
        deadline = started + seconds
        emit("start", seed=seed, budget_s=seconds, dataset_sha256=trainer.dataset_sha256,
             learning_rate=trainer.settings["learning_rate"], training_data_split="train")
        if arm == "parent":
            trainer.initialize_examples(min(deadline, time.monotonic() + trainer.allocation["initial_data_seconds"]))
        steps = 0
        while trainer.active(deadline):
            if arm in ("parent", "patternboost"):
                trainer.supervised_step()
                if (trainer.counters["supervised_updates"] % trainer.settings["refresh_every_updates"] == 0
                        and trainer.active(deadline)):
                    trainer.refresh(deadline)
            else:
                trainer.algorithm_step(deadline)
            steps += 1
            if arm != "parent" and steps % DIAGNOSTIC_EVERY == 0:
                trainer.diagnose(deadline)
        elapsed = time.monotonic() - started
        segment = {key: value - initial_counters[key] for key, value in trainer.counters.items()}
        updates = sum(segment.get(key, 0) for key in ("supervised_updates", "rl_updates", "dpo_updates"))
        trainer.final_status = ("interrupted" if trainer.stopped else
                                "complete" if updates and elapsed >= seconds else "insufficient_training")
        trainer.segment_training_seconds = min(seconds, elapsed) if trainer.stopped else seconds
        base = getattr(trainer, "base_training_seconds", 0.0)
        trainer.training_seconds = base + trainer.segment_training_seconds
        trainer.measured_training_seconds = getattr(trainer, "base_measured_seconds", 0.0) + elapsed
        if arm != "parent":
            trainer.additional_training_seconds = trainer.segment_training_seconds
        io_started = time.monotonic()
        saved = trainer.checkpoint()
        validate_final_checkpoint(saved, trainer.dataset_sha256, seed, arm, require_complete=False)
        target = output / "final.pt"
        temporary = output / "final.pt.tmp"
        trainer.torch.save(saved, temporary)
        os.replace(temporary, target)
        digest = file_sha256(target)
        io_seconds = time.monotonic() - io_started
        result = {key: saved[key] for key in (
            "schema", "kind", "role", "arm", "status", "seed", "threads", "model_config",
            "dataset_sha256", "training_seconds", "segment_training_seconds", "measured_training_seconds",
            "parent_file_sha256", "parent_state_sha256", "initial_state_sha256", "state_sha256",
            "selection", "training_data_split", "final_test_enabled", "test_evaluated", "final_campaign",
            "lineage", "handoff", "counters", "timing")}
        result.update(budget_s=seconds, compute_elapsed_s=elapsed, checkpoint_io_s=io_seconds,
                      elapsed_s=time.monotonic() - started, base_training_seconds=base,
                      additional_training_seconds=trainer.segment_training_seconds,
                      training_settings=trainer.settings, segment_counters=segment,
                      final_state_sha256=saved["state_sha256"], checkpoint_path=str(target.resolve()),
                      checkpoint_sha256=digest,
                      checkpoint_records=[{"filename": "final.pt", "file_sha256": digest,
                                           "state_sha256": saved["state_sha256"], "selected": False,
                                           "training_seconds": trainer.training_seconds,
                                           "checkpoint_write_s": io_seconds}],
                      diagnostics=getattr(trainer, "diagnostics", {}),
                      reference_state_sha256=saved.get("reference_state_sha256"),
                      timing_note="Initialization, data generation, reference rollouts and diagnostics count. Final checkpoint I/O is separate. In-flight operations can overrun the nominal cutoff.")
        atomic_json(output / "result.json", result)
        emit("finished", status=trainer.final_status)
        return result
    except Exception as exc:
        atomic_json(output / "result.json", {"schema": 1, "status": "failed", "arm": arm, "seed": seed,
                    "budget_s": seconds, "elapsed_s": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}", "training_data_split": "train",
                    "test_evaluated": False, "final_test_enabled": False})
        raise
    finally:
        handle.close()
        for number, previous in handlers.items():
            signal.signal(number, previous)


def train_parent(data, config, output, seed, seconds, threads):
    return _run(data, config, output, seed, seconds, threads, "parent")


def train_endpoint(data, config, output, seed, arm, seconds, threads, parent_checkpoint=None):
    if arm not in ENDPOINT_ARMS:
        raise ValueError("Unknown final endpoint arm")
    return _run(data, config, output, seed, seconds, threads, arm, parent_checkpoint)
