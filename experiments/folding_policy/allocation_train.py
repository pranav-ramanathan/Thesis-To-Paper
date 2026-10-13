"""Shared-prefix allocation pilot; complete resumable training state, no test access.

Training clock includes initialization and example generation. Checkpoint writing
is excluded from that clock and reported separately. Operations already underway
may overrun a stage boundary; nominal and measured lineage times are both saved.
This module deliberately keeps the original pilot's model and update equations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

try:
    from . import geometry
    from .train import (DEFAULTS, ElitePool, atomic_json, check_splits, greedy_complete,
                        group_advantages, outcome, state_hash, stratified_rows)
except ImportError:
    import geometry
    from train import (DEFAULTS, ElitePool, atomic_json, check_splits, greedy_complete,
                       group_advantages, outcome, state_hash, stratified_rows)

MODES = ("patternboost", "rl_scratch", "rl_branch")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_hash(value):
    """Stable hash for optimizer/RNG state; tensors are compared byte-for-byte."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


class AllocationTrainer:
    """Training state with explicit steps, enabling exact continuation checks."""
    def __init__(self, data_path, seed, threads, config=None, emit=None):
        import torch
        try:
            from .model import Policy, sample, supervised_loss, trajectory_log_probs
        except ImportError:
            from model import Policy, sample, supervised_loss, trajectory_log_probs
        self.torch = torch
        self.sample, self.supervised_loss, self.log_probs = sample, supervised_loss, trajectory_log_probs
        self.seed, self.threads = seed, threads
        self.config = copy.deepcopy(config or {})
        self.settings = {**DEFAULTS, **self.config.get("training", {})}
        self.allocation = {"milestones_seconds": [3600, 5400, 7200],
                           "initial_data_seconds": 900, **self.config.get("allocation", {})}
        if threads < 1 or self.settings["grpo_kl_beta"] != 0:
            raise ValueError("Positive CPU threads and the declared KL-beta-zero objective are required")
        if self.settings["trap_penalty"] >= 0 or self.settings["group_size"] < 2:
            raise ValueError("Negative trap penalty and a group size of at least two are required")
        for name in ("batch_size", "elite_size", "temperature", "learning_rate",
                     "gradient_clip", "refresh_every_updates", "grpo_epochs"):
            if self.settings[name] <= 0:
                raise ValueError(f"Invalid positive training setting: {name}")
        for name in ("regrow_probability", "greedy_completion_probability"):
            if not 0 <= self.settings[name] <= 1:
                raise ValueError(f"Invalid probability: {name}")
        if not 0 < self.settings["grpo_clip"] < 1 or self.allocation["initial_data_seconds"] <= 0:
            raise ValueError("Invalid clipping or initial-example budget")
        raw = Path(data_path).read_bytes()
        self.dataset_sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        check_splits(self.data)
        if "split_sha256" in self.data:
            try:
                from .dataset import validate_dataset
            except ImportError:
                from dataset import validate_dataset
            validate_dataset(self.data)
        torch.set_num_threads(threads)
        random.seed(seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.policy = Policy(self.config.get("model", {})).cpu()
        if float(self.policy.config.get("dropout", 0)) != 0:
            raise ValueError("Dropout must stay zero for the original GRPO ratios")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.settings["learning_rate"])
        self.pool = ElitePool(self.settings["elite_size"], geometry)
        self.counters = {key: 0 for key in (
            "generated_folds", "valid_folds", "interrupted_folds", "supervised_examples",
            "supervised_updates", "rl_updates", "rl_groups", "zero_advantage_groups",
            "retained_candidates", "greedy_completions")}
        self.timing = {key: 0.0 for key in ("initialization_s", "data_generation_s",
                                           "supervised_s", "candidate_generation_s", "rl_s")}
        self.coverage = {row["sequence"]: {"attempts": 0, "valid_records": 0}
                         for row in self.data["train"]}
        self.emit = emit or (lambda *_args, **_kwargs: None)
        self.data_initialized = False
        self.handoff = None
        self.condition = "patternboost"
        self.mode = "patternboost"
        self.stage = "patternboost"
        self.training_seconds = 0.0
        self.measured_training_seconds = 0.0
        self.lineage = {"root_seed": seed, "parent": None}
        self.stopped = False

    def active(self, deadline):
        return not self.stopped and time.monotonic() < deadline

    def observe(self, rows):
        self.counters["generated_folds"] += len(rows)
        self.counters["valid_folds"] += sum(bool(row["complete"]) for row in rows)
        self.counters["interrupted_folds"] += sum(bool(row.get("interrupted")) for row in rows)

    def initialize_examples(self, deadline):
        if self.data_initialized:
            raise ValueError("Initial examples must not be regenerated in a shared trajectory")
        begin = time.monotonic()
        schedule = stratified_rows(self.data["train"], self.rng)
        index = 0
        while self.active(deadline):
            seq = schedule[index % len(schedule)]["sequence"]
            self.coverage[seq]["attempts"] += 1
            row = geometry.greedy(seq, rng=self.rng, attempts=1, deadline=deadline)
            self.counters["generated_folds"] += 1
            if row is not None and row["complete"]:
                row["source"] = "randomized_greedy"
                self.counters["valid_folds"] += 1
                self.coverage[seq]["valid_records"] += 1
                self.counters["retained_candidates"] += int(self.pool.add(row))
            elif time.monotonic() >= deadline:
                self.counters["interrupted_folds"] += 1
            index += 1
            if index >= len(schedule) and len(self.pool) >= len(schedule) * self.settings["elite_size"]:
                break
        self.data_initialized = True
        self.timing["data_generation_s"] += time.monotonic() - begin
        self.emit("greedy_data", attempts=index, elite_count=len(self.pool))
        if not len(self.pool):
            raise RuntimeError("No valid examples within the fixed initial-data budget")

    def supervised_step(self):
        begin = time.monotonic()
        rows = self.pool.draw(self.rng, self.settings["batch_size"])
        if not rows:
            raise RuntimeError("Cannot supervise an empty elite pool")
        augmented = [geometry.augment(row["sequence"], row["positions"], self.rng) for row in rows]
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.supervised_loss(self.policy, augmented)
        if not self.torch.isfinite(loss):
            raise RuntimeError("Nonfinite supervised loss")
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings["gradient_clip"])
        self.optimizer.step()
        self.counters["supervised_updates"] += 1
        self.counters["supervised_examples"] += len(rows)
        self.timing["supervised_s"] += time.monotonic() - begin
        self.emit("supervised_update", loss=float(loss.detach()), examples=len(rows), elite_count=len(self.pool))
        return float(loss.detach())

    def refresh(self, deadline):
        begin = time.monotonic()
        sequences = [self.rng.choice(self.data["train"])["sequence"] for _ in range(self.settings["batch_size"])]
        prefixes = []
        for seq in sequences:
            elites = self.pool.by_sequence.get(seq, [])
            if elites and self.rng.random() < self.settings["regrow_probability"]:
                fold = self.rng.choice(elites)["positions"]
                prefixes.append(fold[:self.rng.randrange(2, len(fold))])
            else:
                prefixes.append(None)
        self.policy.eval()
        rows = self.sample(self.policy, sequences, generator=self.generator,
                           temperature=self.settings["temperature"], prefixes=prefixes, deadline=deadline)
        for row in rows:
            row["source"] = "model_or_model_suffix_regrowth"
        self.observe(rows)
        retained = sum(int(self.pool.add(row)) for row in rows if not row.get("interrupted"))
        completions = []
        for row in rows:
            if not self.active(deadline):
                break
            if len(row["positions"]) > 2 and self.rng.random() < self.settings["greedy_completion_probability"]:
                prefix = row["positions"][:self.rng.randrange(2, len(row["positions"]))]
                completed = greedy_complete(row["sequence"], prefix, self.rng, geometry, deadline)
                completed["source"] = "model_prefix_greedy_completion"
                completions.append(completed)
        self.observe(completions)
        self.counters["greedy_completions"] += len(completions)
        retained += sum(int(self.pool.add(row)) for row in completions if not row.get("interrupted"))
        self.counters["retained_candidates"] += retained
        self.timing["candidate_generation_s"] += time.monotonic() - begin
        self.emit("pool_refresh", generated=len(rows), greedy_completions=len(completions),
                  retained=retained, elite_count=len(self.pool))

    def reinforcement_step(self, deadline):
        begin = time.monotonic()
        try:
            seq = self.rng.choice(self.data["train"])["sequence"]
            self.policy.eval()
            rows = self.sample(self.policy, [seq] * self.settings["group_size"],
                               generator=self.generator, temperature=self.settings["temperature"], deadline=deadline)
            self.observe(rows)
            rows = [row for row in rows if not row.get("interrupted")]
            if len(rows) < 2 or not self.active(deadline):
                return False
            rewards = [outcome(row, self.settings["trap_penalty"]) for row in rows]
            advantages = group_advantages(rewards)
            self.counters["rl_groups"] += 1
            if not any(advantages):
                self.counters["zero_advantage_groups"] += 1
                self.emit("rl_group", rewards=rewards, zero_advantage=True, updated=False)
                return False
            torch = self.torch
            with torch.no_grad():
                old, taken = self.log_probs(self.policy, rows, temperature=self.settings["temperature"])
            if not bool(taken.any()):
                return False
            advantage = torch.tensor(advantages, dtype=old.dtype, device=old.device).unsqueeze(1)
            updated, loss_value = False, None
            for _ in range(self.settings["grpo_epochs"]):
                if not self.active(deadline):
                    break
                self.policy.train()
                self.optimizer.zero_grad(set_to_none=True)
                current, current_taken = self.log_probs(self.policy, rows, temperature=self.settings["temperature"])
                if not torch.equal(taken, current_taken):
                    raise RuntimeError("GRPO masks changed on identical trajectories")
                ratio = (current - old).exp()
                clipped = ratio.clamp(1 - self.settings["grpo_clip"], 1 + self.settings["grpo_clip"])
                surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                loss = -((surrogate * taken).sum(1) / taken.sum(1).clamp_min(1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite GRPO loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings["gradient_clip"])
                self.optimizer.step()
                self.counters["rl_updates"] += 1
                updated, loss_value = True, float(loss.detach())
            self.emit("rl_group", rewards=rewards, zero_advantage=False, updated=updated,
                      valid=sum(row["complete"] for row in rows), loss=loss_value)
            return updated
        finally:
            self.timing["rl_s"] += time.monotonic() - begin

    def checkpoint(self):
        rng_states = {"python_global": random.getstate(), "python_local": self.rng.getstate(),
                      "torch_global": self.torch.get_rng_state().clone(),
                      "torch_generator": self.generator.get_state().clone()}
        return {"schema": 1, "kind": "allocation_training_state",
                "model_config": dict(self.policy.config), "model_state": copy.deepcopy(self.policy.state_dict()),
                "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
                "rng_states": rng_states, "elite_pool": copy.deepcopy(self.pool.by_sequence),
                "counters": dict(self.counters), "timing": dict(self.timing),
                "data_coverage": copy.deepcopy(self.coverage), "data_initialized": self.data_initialized,
                "dataset_sha256": self.dataset_sha256, "config": copy.deepcopy(self.config),
                "effective_training_settings": dict(self.settings),
                "condition": self.condition, "mode": self.mode, "seed": self.seed, "threads": self.threads,
                "stage": self.stage, "training_seconds": self.training_seconds,
                "measured_training_seconds": self.measured_training_seconds,
                "state_sha256": state_hash(self.policy.state_dict()),
                "optimizer_sha256": nested_hash(self.optimizer.state_dict()),
                "lineage": copy.deepcopy(self.lineage), "handoff": copy.deepcopy(self.handoff),
                "policy_training_mode": self.policy.training,
                "selection": {"split": "none", "validated": False, "selected": False},
                "final_test_enabled": False, "test_evaluated": False}

    def restore(self, checkpoint):
        if checkpoint.get("schema") != 1 or checkpoint.get("kind") != "allocation_training_state":
            raise ValueError("Expected a complete allocation training checkpoint")
        for field, expected in (("seed", self.seed), ("threads", self.threads),
                                ("dataset_sha256", self.dataset_sha256), ("config", self.config),
                                ("model_config", self.policy.config),
                                ("effective_training_settings", self.settings)):
            if checkpoint.get(field) != expected:
                raise ValueError(f"Resume {field} mismatch")
        if checkpoint.get("final_test_enabled") is not False or checkpoint.get("test_evaluated") is not False:
            raise ValueError("Resume must not include final-test evaluation")
        if state_hash(checkpoint["model_state"]) != checkpoint.get("state_sha256"):
            raise ValueError("Resume model hash mismatch")
        if nested_hash(checkpoint["optimizer_state"]) != checkpoint.get("optimizer_sha256"):
            raise ValueError("Resume optimizer hash mismatch")
        train_sequences = set(self.coverage)
        for sequence, rows in checkpoint["elite_pool"].items():
            if sequence not in train_sequences or len(rows) > self.settings["elite_size"]:
                raise ValueError("Resume elite pool contains an invalid sequence or size")
            keys = set()
            for row in rows:
                if row["sequence"] != sequence or geometry.validate(sequence, row["positions"]) != row["contacts"]:
                    raise ValueError("Resume elite pool contains an invalid fold")
                key = geometry.canonical_key(row["positions"])
                if key in keys:
                    raise ValueError("Resume elite pool contains symmetry duplicates")
                keys.add(key)
        self.policy.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.pool.by_sequence = copy.deepcopy(checkpoint["elite_pool"])
        self.counters, self.timing = dict(checkpoint["counters"]), dict(checkpoint["timing"])
        self.coverage = copy.deepcopy(checkpoint["data_coverage"])
        self.data_initialized = checkpoint["data_initialized"]
        self.condition, self.stage = checkpoint["condition"], checkpoint["stage"]
        self.mode = checkpoint["mode"]
        self.training_seconds = checkpoint["training_seconds"]
        self.measured_training_seconds = checkpoint["measured_training_seconds"]
        self.lineage, self.handoff = copy.deepcopy(checkpoint["lineage"]), copy.deepcopy(checkpoint["handoff"])
        self.policy.train(checkpoint["policy_training_mode"])
        random.setstate(checkpoint["rng_states"]["python_global"])
        self.rng.setstate(checkpoint["rng_states"]["python_local"])
        self.torch.set_rng_state(checkpoint["rng_states"]["torch_global"])
        self.generator.set_state(checkpoint["rng_states"]["torch_generator"])

    def begin_rl_branch(self, parent_path, parent_sha256):
        if self.condition != "patternboost" or self.stage != "patternboost":
            raise ValueError("RL branches must start at a PatternBoost stage")
        if self.counters["supervised_updates"] == 0:
            raise ValueError("RL branches need a trained PatternBoost parent")
        before = state_hash(self.policy.state_dict())
        optimizer_hash = nested_hash(self.optimizer.state_dict())
        self.condition, self.stage = "patternboost_rl", "rl"
        self.mode = "rl_branch"
        self.lineage = {"root_seed": self.seed, "parent": str(Path(parent_path).resolve()),
                        "parent_file_sha256": parent_sha256, "pretraining_seconds": self.training_seconds,
                        "pretraining_measured_seconds": self.measured_training_seconds,
                        "parent_state_sha256": before, "parent_optimizer_sha256": optimizer_hash}
        self.handoff = {"pretraining_sha256": before, "rl_initial_sha256": before,
                        "pretraining_optimizer_sha256": optimizer_hash,
                        "rl_initial_optimizer_sha256": nested_hash(self.optimizer.state_dict()),
                        "optimizer_state_retained": True, "rng_state_retained": True,
                        "elite_pool_retained": True, "supervised_updates": self.counters["supervised_updates"]}
        self.emit("stage_handoff", **self.handoff)


def run(data_path, mode, seed, seconds, threads, output, config=None, resume=None):
    started = time.monotonic()
    if mode not in MODES or seconds <= 0 or threads < 1:
        raise ValueError("Invalid allocation training request")
    if (mode == "rl_branch") != (resume is not None):
        raise ValueError("Only RL branch mode requires a parent checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Training output must be a fresh empty directory")
    excluded_io = 0.0
    records = []
    handle = (output / "events.jsonl").open("w", buffering=1)
    trainer = None
    def emit(kind, **fields):
        handle.write(json.dumps({"type": kind, "elapsed_s": time.monotonic() - started,
                                 "stage": trainer.stage if trainer is not None else "initialization",
                                 **fields}, sort_keys=True) + "\n")
    def local_compute():
        return time.monotonic() - started - excluded_io
    handlers = {}
    base_nominal, base_measured = 0.0, 0.0
    try:
        trainer = AllocationTrainer(data_path, seed, threads, config=config, emit=emit)
        if resume:
            parent_sha = file_sha256(resume)
            checkpoint = trainer.torch.load(resume, map_location="cpu", weights_only=False)
            trainer.restore(checkpoint)
            if (trainer.training_seconds not in trainer.allocation["milestones_seconds"][:-1]
                    or abs(trainer.training_seconds + seconds - trainer.allocation["milestones_seconds"][-1]) > 1e-9):
                raise ValueError("Branch parent stage and RL budget must match the declared allocation")
            trainer.begin_rl_branch(resume, parent_sha)
            base_nominal, base_measured = trainer.training_seconds, trainer.measured_training_seconds
        else:
            trainer.condition = mode
            trainer.mode = mode
            trainer.stage = "rl" if mode == "rl_scratch" else "patternboost"
        initial_hash = state_hash(trainer.policy.state_dict())
        initial_counters = dict(trainer.counters)
        trainer.timing["initialization_s"] += local_compute()
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda *_: setattr(trainer, "stopped", True))
        emit("start", mode=mode, seed=seed, budget_s=seconds, base_training_seconds=base_nominal,
             dataset_sha256=trainer.dataset_sha256)
        def save(name, nominal):
            nonlocal excluded_io
            trainer.training_seconds = nominal
            trainer.measured_training_seconds = base_measured + local_compute()
            begin = time.monotonic()
            value = trainer.checkpoint()
            target = output / name
            temporary = target.with_name(target.name + ".tmp")
            trainer.torch.save(value, temporary)
            os.replace(temporary, target)
            digest = file_sha256(target)
            duration = time.monotonic() - begin
            excluded_io += duration
            record = {"filename": name, "stage": trainer.stage, "training_seconds": nominal,
                      "measured_training_seconds": value["measured_training_seconds"],
                      "state_sha256": value["state_sha256"], "state_hash": value["state_sha256"],
                      "optimizer_sha256": value["optimizer_sha256"], "file_sha256": digest,
                      "checkpoint_write_s": duration, "selected": False}
            records.append(record)
            emit("checkpoint", **record)
        def deadline_at(local_limit):
            return started + excluded_io + local_limit
        if mode == "patternboost":
            milestones = trainer.allocation["milestones_seconds"]
            if (not milestones or any(not isinstance(value, (int, float)) or value <= 0 for value in milestones)
                    or sorted(set(milestones)) != milestones or milestones[-1] != seconds
                    or trainer.allocation["initial_data_seconds"] >= milestones[0]):
                raise ValueError("PatternBoost milestones must increase to the total budget, after initial-data allocation")
            data_deadline = min(deadline_at(seconds), time.monotonic() + trainer.allocation["initial_data_seconds"])
            trainer.initialize_examples(data_deadline)
            for milestone in milestones:
                limit = deadline_at(milestone)
                while trainer.active(limit):
                    trainer.supervised_step()
                    if trainer.counters["supervised_updates"] % trainer.settings["refresh_every_updates"] == 0 and trainer.active(limit):
                        trainer.refresh(limit)
                if trainer.stopped:
                    break
                save(f"stage_{milestone:g}.pt", float(milestone))
        else:
            limit = deadline_at(seconds)
            while trainer.active(limit):
                trainer.reinforcement_step(limit)
        status = "interrupted" if trainer.stopped else "complete"
        if mode == "patternboost" and trainer.counters["supervised_updates"] == 0:
            status = "insufficient_training"
        if mode != "patternboost" and trainer.counters["rl_updates"] == initial_counters["rl_updates"]:
            status = "insufficient_training" if not trainer.stopped else "interrupted"
        nominal = base_nominal + (min(seconds, local_compute()) if trainer.stopped else seconds)
        save("final.pt", nominal)
        result = {"schema": 1, "status": status, "mode": mode, "condition": trainer.condition,
                  "seed": seed, "threads": threads, "budget_s": seconds,
                  "training_seconds": nominal, "measured_training_seconds": trainer.measured_training_seconds,
                  "compute_elapsed_s": local_compute(), "elapsed_s": time.monotonic() - started,
                  "checkpoint_io_s": excluded_io, "dataset_sha256": trainer.dataset_sha256,
                  "model_config": trainer.policy.config, "training_settings": trainer.settings,
                  "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(trainer.policy.state_dict()),
                  "checkpoint_records": records, "lineage": trainer.lineage, "handoff": trainer.handoff,
                  "counters": trainer.counters,
                  "segment_counters": {key: value - initial_counters[key] for key, value in trainer.counters.items()},
                  "timing": trainer.timing, "elite_count": len(trainer.pool),
                  "elite_sequence_count": len(trainer.pool.by_sequence),
                  "training_sequences": len(trainer.data["train"]), "data_coverage": trainer.coverage,
                  "selection": {"split": "none", "selected": False},
                  "test_evaluated": False, "final_test_enabled": False,
                  "limitations": ["Checkpoints are prespecified endpoints, not validation-selected policies.",
                                  "Nominal lineage training time and measured training time are both retained.",
                                  "Initialization and data generation count; checkpoint serialization is reported separately.",
                                  "An ongoing tensor operation can overrun a stage boundary."]}
        atomic_json(output / "result.json", result)
        emit("finished", status=status)
        return result
    except Exception as exc:
        atomic_json(output / "result.json", {"schema": 1, "status": "failed", "mode": mode,
                    "seed": seed, "budget_s": seconds, "elapsed_s": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}", "checkpoint_records": records,
                    "test_evaluated": False, "final_test_enabled": False})
        raise
    finally:
        handle.close()
        for number, previous in handlers.items():
            signal.signal(number, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run(args.data, args.mode, args.seed, args.seconds, args.threads, args.output,
                 config=config, resume=args.resume)
    print(json.dumps({key: result[key] for key in ("status", "mode", "seed", "elapsed_s")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
