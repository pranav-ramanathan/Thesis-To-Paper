"""Fixed-budget fine-tuning from a complete PatternBoost parent.

This is a fixed-configuration screen, not an equally tuned algorithm ranking.
All five arms restore the actor, Adam moments, random states and elite pool.
The sampling policy is unchanged; only the training objective and explicitly
recorded learning rate differ. No validation or final-test folds are sampled.
Initialization, reference rollouts and diagnostics count toward training time.
Final checkpoint serialization is reported separately. An in-flight tensor
operation can finish after the deadline; sampling never accepts interrupted folds.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

try:
    from . import geometry
    from .allocation_train import AllocationTrainer, file_sha256, nested_hash
    from .evaluate import verified_score
    from .train import atomic_json, outcome, state_hash
except ImportError:
    import geometry
    from allocation_train import AllocationTrainer, file_sha256, nested_hash
    from evaluate import verified_score
    from train import atomic_json, outcome, state_hash

ARMS = ("patternboost", "grpo", "dr_grpo", "reinforce", "dpo")
LEARNING_RATES = {arm: (3e-4 if arm in ("patternboost", "grpo") else 3e-5) for arm in ARMS}
DPO_BETA = 0.1
DIAGNOSTIC_EVERY = 20


def dr_grpo_loss(current, old, taken, advantages, max_moves, clip=0.2):
    """Clipped group-relative objective with one global action denominator."""
    import torch
    if max_moves < 1:
        raise ValueError("The global maximum learned-action count must be positive")
    advantage = torch.as_tensor(advantages, dtype=current.dtype, device=current.device).unsqueeze(1)
    ratio = (current - old.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -((surrogate * taken).sum(1) / max_moves).mean()


def reinforce_loss(logp, taken, advantages, max_moves):
    """Sequence log probability with a frozen, action-independent baseline."""
    import torch
    if max_moves < 1:
        raise ValueError("The global maximum learned-action count must be positive")
    advantage = torch.as_tensor(advantages, dtype=logp.dtype, device=logp.device)
    return -((logp * taken).sum(1) * advantage / max_moves).mean()


def dpo_loss(policy_sums, reference_sums, beta=DPO_BETA):
    """Each row is [preferred sequence logp, rejected sequence logp]."""
    import torch.nn.functional as functional
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    relative = policy_sums - reference_sums.detach()
    return -functional.logsigmoid(beta * (relative[:, 0] - relative[:, 1])).mean()


def preference_pairs(rows):
    """Independently score complete same-sequence folds; never prefer a trap.

    Deterministically pair highest with lowest without reusing a trajectory.
    Tied rewards, interrupted folds and malformed witnesses are excluded.
    """
    valid = []
    sequences = {row.get("sequence") for row in rows}
    if len(sequences) > 1:
        raise ValueError("DPO pairs must share one input HP sequence")
    for row in rows:
        if not row.get("complete") or row.get("interrupted"):
            continue
        try:
            verified = verified_score(row["sequence"], row["positions"])
            if verified != row.get("contacts") or outcome(row, -10) != verified:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        valid.append((verified, row))
    valid.sort(key=lambda pair: pair[0])
    return [(valid[-1 - i][1], valid[i][1]) for i in range(len(valid) // 2)
            if valid[-1 - i][0] > valid[i][0]]


def greedy_policy_rollout(policy, sequence, deadline, active=None):
    """Frozen-policy argmax rollout; deterministic first-index tie handling."""
    import torch
    try:
        from .model import encode_states
    except ImportError:
        from model import encode_states
    points = [(0, 0, 0), (1, 0, 0)]
    device = next(policy.parameters()).device
    interrupted = False
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            while len(points) < len(sequence):
                legal = geometry.legal_moves(sequence, points)
                if not legal:
                    break
                if time.monotonic() >= deadline or (active is not None and not active(deadline)):
                    interrupted = True
                    break
                logits = policy(*encode_states([sequence], [points], device))[0]
                if time.monotonic() >= deadline or (active is not None and not active(deadline)):
                    interrupted = True
                    break
                mask = torch.zeros(6, dtype=torch.bool, device=device)
                mask[legal] = True
                action = int(logits.masked_fill(~mask, -torch.inf).argmax())
                points.append(tuple(a + b for a, b in zip(points[-1], geometry.MOVES[action])))
    finally:
        policy.train(was_training)
    row = geometry.trajectory(sequence, points)
    row["interrupted"] = interrupted
    return row


class FineTuneTrainer(AllocationTrainer):
    def __init__(self, data_path, parent_path, arm, seed, threads, learning_rate=None, emit=None):
        import torch
        if arm not in ARMS:
            raise ValueError("Unknown fine-tuning arm")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        super().__init__(data_path, seed, threads, config=parent["config"], emit=emit)
        self.restore(parent)
        if (self.condition != "patternboost" or self.stage != "patternboost"
                or self.mode != "patternboost" or not self.data_initialized
                or self.counters["supervised_updates"] < 1 or self.training_seconds <= 0
                or parent.get("selection", {}).get("selected") is not False):
            raise ValueError("Fine-tuning requires a trained, unselected PatternBoost parent")
        if self.settings["grpo_epochs"] != 1:
            raise ValueError("This fixed screen declares one update epoch per group")
        self.arm = arm
        self.base_training_seconds = self.training_seconds
        self.base_measured_seconds = self.measured_training_seconds
        self.parent_file_sha256 = file_sha256(parent_path)
        self.parent_state_sha256 = state_hash(self.policy.state_dict())
        self.parent_optimizer_sha256 = nested_hash(self.optimizer.state_dict())
        self.initial_counters = dict(self.counters)
        self.initial_timing = dict(self.timing)
        self.reference = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.max_moves = max(len(row["sequence"]) - 2 for row in self.data["train"])
        rate = LEARNING_RATES[arm] if learning_rate is None else learning_rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.settings["learning_rate"] = rate
        self.condition = arm
        self.mode = "finetune"
        self.stage = arm
        self.additional_training_seconds = 0.0
        self.counters.update({"dpo_updates": 0, "preference_updates": 0, "preference_pairs": 0,
                              "no_preference_groups": 0, "reference_rollouts": 0,
                              "reference_complete_folds": 0, "reference_interrupted_folds": 0})
        self.initial_counters = {key: self.initial_counters.get(key, 0) for key in self.counters}
        self.handoff = {"pretraining_sha256": self.parent_state_sha256,
                        "finetune_initial_sha256": self.parent_state_sha256,
                        "pretraining_optimizer_sha256": self.parent_optimizer_sha256,
                        "initial_optimizer_sha256_after_lr_override": nested_hash(self.optimizer.state_dict()),
                        "optimizer_moments_retained": True, "rng_state_retained": True,
                        "elite_pool_retained": True, "learning_rate_override": rate}
        self.lineage = {"root_seed": seed, "parent_file_sha256": self.parent_file_sha256,
                        "parent_state_sha256": self.parent_state_sha256,
                        "pretraining_seconds": self.base_training_seconds,
                        "parent_optimizer_sha256": self.parent_optimizer_sha256}
        self.diagnostics = {"records": 0, "sampled_logp_drift_mean": None,
                            "unique_complete_fold_fraction": None,
                            "sampled_sequence_count": None, "complete_sequence_count": None,
                            "gradient_norm_after_clip": None, "reward_sd": None,
                            "reward_min": None, "reward_max": None,
                            "interpretation": "Sampled current-minus-parent action log probability is not exact KL.",
                            "diversity_note": "Uniqueness is within the sampled batch, not global diversity. PatternBoost can mix input sequences; RL and DPO groups share one input. Fractions across these contexts are not directly comparable."}
        self._last_rows = []
        sampler = self.sample
        def capture(*args, **kwargs):
            rows = sampler(*args, **kwargs)
            self._last_rows = rows
            return rows
        self.sample = capture

    def _apply(self, loss):
        if not self.torch.isfinite(loss):
            raise RuntimeError("Nonfinite fine-tuning loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings["gradient_clip"])
        self.optimizer.step()
        return float(loss.detach())

    def _group(self, deadline):
        seq = self.rng.choice(self.data["train"])["sequence"]
        self.policy.eval()
        rows = self.sample(self.policy, [seq] * self.settings["group_size"], generator=self.generator,
                           temperature=self.settings["temperature"], deadline=deadline)
        self.observe(rows)
        return seq, [row for row in rows if not row.get("interrupted")]

    def algorithm_step(self, deadline):
        if self.arm == "grpo":
            # Preserve the historical update equations, random consumption and Adam state.
            return super().reinforcement_step(deadline)
        begin = time.monotonic()
        try:
            seq, rows = self._group(deadline)
            if len(rows) < 2 or not self.active(deadline):
                return False
            self.counters["rl_groups"] += 1
            rewards = [outcome(row, self.settings["trap_penalty"]) for row in rows]
            if self.arm == "dpo":
                pairs = preference_pairs(rows)
                if not pairs:
                    self.counters["no_preference_groups"] += 1
                    self.emit("dpo_group", rewards=rewards, pairs=0, updated=False)
                    return False
                paired_rows = [row for pair in pairs for row in pair]
                with self.torch.no_grad():
                    reference, reference_taken = self.log_probs(self.reference, paired_rows,
                                                                temperature=self.settings["temperature"])
                if not self.active(deadline):
                    return False
                self.policy.train()
                current, taken = self.log_probs(self.policy, paired_rows, temperature=self.settings["temperature"])
                if not self.torch.equal(taken, reference_taken):
                    raise RuntimeError("DPO policy and reference masks differ")
                if not self.active(deadline):
                    return False
                loss = dpo_loss((current * taken).sum(1).reshape(-1, 2),
                                (reference * taken).sum(1).reshape(-1, 2))
                value = self._apply(loss)
                self.counters["dpo_updates"] += 1
                self.counters["preference_updates"] += 1
                self.counters["preference_pairs"] += len(pairs)
                self.emit("dpo_group", rewards=rewards, pairs=len(pairs), loss=value, updated=True)
                return True
            baseline_reward = None
            if self.arm == "reinforce":
                baseline = greedy_policy_rollout(self.reference, seq, deadline, self.active)
                self.counters["reference_rollouts"] += 1
                self.counters["reference_complete_folds"] += int(baseline["complete"])
                self.counters["reference_interrupted_folds"] += int(baseline["interrupted"])
                if baseline["interrupted"] or not self.active(deadline):
                    return False
                baseline_reward = outcome(baseline, self.settings["trap_penalty"])
                advantages = [reward - baseline_reward for reward in rewards]
            else:
                mean = statistics.mean(rewards)
                advantages = [reward - mean for reward in rewards]
            if not any(advantages):
                self.counters["zero_advantage_groups"] += 1
                self.emit("rl_group", rewards=rewards, zero_advantage=True, updated=False,
                          reference_reward=baseline_reward)
                return False
            old = None
            if self.arm == "dr_grpo":
                with self.torch.no_grad():
                    old, _ = self.log_probs(self.policy, rows, temperature=self.settings["temperature"])
            if not self.active(deadline):
                return False
            self.policy.train()
            current, taken = self.log_probs(self.policy, rows, temperature=self.settings["temperature"])
            if not bool(taken.any()) or not self.active(deadline):
                return False
            loss = (dr_grpo_loss(current, old, taken, advantages, self.max_moves, self.settings["grpo_clip"])
                    if self.arm == "dr_grpo" else reinforce_loss(current, taken, advantages, self.max_moves))
            value = self._apply(loss)
            self.counters["rl_updates"] += 1
            self.emit("rl_group", rewards=rewards, zero_advantage=False, updated=True,
                      valid=sum(row["complete"] for row in rows), loss=value, reference_reward=baseline_reward)
            return True
        finally:
            self.timing["rl_s"] += time.monotonic() - begin

    def diagnose(self, deadline):
        """A fixed cadence observes existing training samples without RNG draws."""
        rows = [row for row in self._last_rows if not row.get("interrupted")]
        if not rows or not self.active(deadline):
            return
        rewards = [outcome(row, self.settings["trap_penalty"]) for row in rows]
        complete = [row for row in rows if row["complete"]]
        unique = {(row["sequence"], geometry.canonical_key(row["positions"])) for row in complete}
        gradient = math.sqrt(sum(float(parameter.grad.detach().square().sum())
                                 for parameter in self.policy.parameters() if parameter.grad is not None))
        self.policy.eval()
        with self.torch.no_grad():
            current, taken = self.log_probs(self.policy, rows, temperature=self.settings["temperature"])
            if not self.active(deadline):
                return
            reference, reference_taken = self.log_probs(self.reference, rows,
                                                        temperature=self.settings["temperature"])
        if not self.torch.equal(taken, reference_taken):
            raise RuntimeError("Diagnostic masks differ")
        drift = float(((current - reference) * taken).sum() / taken.sum().clamp_min(1))
        self.diagnostics.update(records=self.diagnostics["records"] + 1,
                                sampled_logp_drift_mean=drift,
                                unique_complete_fold_fraction=len(unique) / len(complete) if complete else None,
                                sampled_sequence_count=len({row["sequence"] for row in rows}),
                                complete_sequence_count=len({row["sequence"] for row in complete}),
                                gradient_norm_after_clip=gradient, reward_sd=statistics.pstdev(rewards),
                                reward_min=min(rewards), reward_max=max(rewards))
        self.emit("training_diagnostics", **self.diagnostics, complete=len(complete), sampled=len(rows))

    def checkpoint(self):
        saved = super().checkpoint()
        saved.update(kind="finetune_training_state", arm=self.arm,
                     base_training_seconds=self.base_training_seconds,
                     additional_training_seconds=self.additional_training_seconds,
                     parent_file_sha256=self.parent_file_sha256,
                     parent_state_sha256=self.parent_state_sha256,
                     reference_state_sha256=state_hash(self.reference.state_dict()),
                     reference_state=copy.deepcopy(self.reference.state_dict()),
                     global_max_learned_moves=self.max_moves, dpo_beta=DPO_BETA,
                     diagnostics=copy.deepcopy(self.diagnostics),
                     segment_counters={key: value - self.initial_counters[key]
                                       for key, value in self.counters.items()},
                     initial_timing=dict(self.initial_timing),
                     segment_timing={key: value - self.initial_timing.get(key, 0.0)
                                     for key, value in self.timing.items()},
                     timing_note="timing is cumulative parent plus new training; segment_timing contains this branch only. Phase timings exclude diagnostics and other framework overhead included in compute_elapsed_s.",
                     initial_counters=dict(self.initial_counters))
        return saved


def validate_finetune_checkpoint(saved, data_hash, seed=None, arm=None, parent_sha=None):
    """Validate provenance and tensor hashes before inference or reporting."""
    if saved.get("schema") != 1 or saved.get("kind") != "finetune_training_state":
        raise ValueError("Expected a complete fine-tuning training checkpoint")
    if saved.get("dataset_sha256") != data_hash:
        raise ValueError("Fine-tuning dataset mismatch")
    if saved.get("arm") not in ARMS or saved.get("condition") != saved.get("arm"):
        raise ValueError("Fine-tuning arm mismatch")
    if saved.get("mode") != "finetune" or saved.get("stage") != saved.get("arm"):
        raise ValueError("Fine-tuning mode/stage mismatch")
    for field, expected in (("seed", seed), ("arm", arm), ("parent_file_sha256", parent_sha)):
        if expected is not None and saved.get(field) != expected:
            raise ValueError(f"Fine-tuning {field} mismatch")
    for field, hash_field in (("model_state", "state_sha256"), ("reference_state", "reference_state_sha256")):
        if field not in saved or state_hash(saved[field]) != saved.get(hash_field):
            raise ValueError(f"Fine-tuning {field} hash mismatch")
    if nested_hash(saved.get("optimizer_state")) != saved.get("optimizer_sha256"):
        raise ValueError("Fine-tuning optimizer hash mismatch")
    if (saved.get("reference_state_sha256") != saved.get("parent_state_sha256")
            or saved.get("handoff", {}).get("finetune_initial_sha256") != saved.get("parent_state_sha256")):
        raise ValueError("Fine-tuning parent/reference mismatch")
    for field in ("base_training_seconds", "additional_training_seconds", "training_seconds"):
        value = saved.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid fine-tuning {field}")
    if (saved["base_training_seconds"] <= 0
            or abs(saved["training_seconds"] - saved["base_training_seconds"] - saved["additional_training_seconds"]) > 1e-6):
        raise ValueError("Fine-tuning lineage duration mismatch")
    if (saved.get("selection", {}).get("selected") is not False
            or saved.get("test_evaluated") is not False or saved.get("final_test_enabled") is not False):
        raise ValueError("Fine-tuning must use fixed endpoints without final-test evaluation")
    if not all(field in saved for field in ("rng_states", "elite_pool", "counters", "model_config", "initial_counters")):
        raise ValueError("Incomplete fine-tuning state")
    return saved


def run(data_path, parent_path, arm, seed, seconds, threads, output, learning_rate=None):
    started = time.monotonic()
    if arm not in ARMS or not math.isfinite(seconds) or seconds <= 0 or threads < 1:
        raise ValueError("Invalid fine-tuning request")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Fine-tuning output must be a fresh empty directory")
    handle = (output / "events.jsonl").open("w", buffering=1)
    handlers, records = {}, []
    trainer = None
    def emit(kind, **fields):
        handle.write(json.dumps({"type": kind, "elapsed_s": time.monotonic() - started,
                                 "arm": arm, **fields}, sort_keys=True) + "\n")
    try:
        trainer = FineTuneTrainer(data_path, parent_path, arm, seed, threads, learning_rate, emit)
        trainer.timing["initialization_s"] += time.monotonic() - started
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda *_: setattr(trainer, "stopped", True))
        deadline = started + seconds
        emit("start", seed=seed, budget_s=seconds, base_training_seconds=trainer.base_training_seconds,
             dataset_sha256=trainer.dataset_sha256, parent_file_sha256=trainer.parent_file_sha256,
             parent_state_sha256=trainer.parent_state_sha256, learning_rate=trainer.settings["learning_rate"],
             global_max_learned_moves=trainer.max_moves, handoff=trainer.handoff)
        steps = 0
        while trainer.active(deadline):
            if arm == "patternboost":
                trainer.supervised_step()
                if (trainer.counters["supervised_updates"] % trainer.settings["refresh_every_updates"] == 0
                        and trainer.active(deadline)):
                    trainer.refresh(deadline)
            else:
                trainer.algorithm_step(deadline)
            steps += 1
            if steps % DIAGNOSTIC_EVERY == 0:
                trainer.diagnose(deadline)
        compute_elapsed = time.monotonic() - started
        segment = {key: value - trainer.initial_counters[key] for key, value in trainer.counters.items()}
        updates = sum(segment[key] for key in ("supervised_updates", "rl_updates", "dpo_updates"))
        status = ("interrupted" if trainer.stopped else
                  ("complete" if updates > 0 and compute_elapsed >= seconds else "insufficient_training"))
        trainer.additional_training_seconds = min(seconds, compute_elapsed) if trainer.stopped else seconds
        trainer.training_seconds = trainer.base_training_seconds + trainer.additional_training_seconds
        trainer.measured_training_seconds = trainer.base_measured_seconds + compute_elapsed
        io_started = time.monotonic()
        saved = trainer.checkpoint()
        validate_finetune_checkpoint(saved, trainer.dataset_sha256, seed, arm, trainer.parent_file_sha256)
        temporary = output / "final.pt.tmp"
        trainer.torch.save(saved, temporary)
        os.replace(temporary, output / "final.pt")
        digest = file_sha256(output / "final.pt")
        io_seconds = time.monotonic() - io_started
        records.append({"filename": "final.pt", "file_sha256": digest,
                        "state_sha256": saved["state_sha256"], "training_seconds": trainer.training_seconds,
                        "additional_training_seconds": trainer.additional_training_seconds,
                        "checkpoint_write_s": io_seconds, "selected": False})
        emit("checkpoint", **records[-1])
        result = {"schema": 1, "status": status, "arm": arm, "condition": arm, "seed": seed,
                  "threads": threads, "budget_s": seconds,
                  "base_training_seconds": trainer.base_training_seconds,
                  "additional_training_seconds": trainer.additional_training_seconds,
                  "training_seconds": trainer.training_seconds,
                  "measured_training_seconds": trainer.measured_training_seconds,
                  "compute_elapsed_s": compute_elapsed, "checkpoint_io_s": io_seconds,
                  "elapsed_s": time.monotonic() - started, "dataset_sha256": trainer.dataset_sha256,
                  "parent_file_sha256": trainer.parent_file_sha256,
                  "parent_state_sha256": trainer.parent_state_sha256,
                  "initial_state_sha256": trainer.parent_state_sha256,
                  "final_state_sha256": saved["state_sha256"], "reference_state_sha256": saved["reference_state_sha256"],
                  "model_config": trainer.policy.config, "training_settings": trainer.settings,
                  "global_max_learned_moves": trainer.max_moves, "dpo_beta": DPO_BETA,
                  "checkpoint_records": records, "counters": trainer.counters, "segment_counters": segment,
                  "diagnostics": trainer.diagnostics, "timing": trainer.timing,
                  "initial_timing": saved["initial_timing"], "segment_timing": saved["segment_timing"],
                  "timing_note": saved["timing_note"],
                  "handoff": trainer.handoff, "lineage": trainer.lineage,
                  "selection": {"split": "none", "selected": False},
                  "test_evaluated": False, "final_test_enabled": False,
                  "limitations": ["Fixed configurations have different learning rates; no equally tuned algorithm ranking is claimed.",
                                  "All initialization, reference rollouts, candidate generation and diagnostics count toward training.",
                                  "Checkpoint serialization is reported separately; in-flight operations may overrun the cutoff.",
                                  "Dr. GRPO and REINFORCE use one global maximum training action count, not per-fold normalization.",
                                  "Sampled log probability drift is not an exact KL divergence."]}
        atomic_json(output / "result.json", result)
        emit("finished", status=status)
        return result
    except Exception as exc:
        atomic_json(output / "result.json", {"schema": 1, "status": "failed", "arm": arm, "seed": seed,
                    "budget_s": seconds, "elapsed_s": time.monotonic() - started,
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
    parser.add_argument("--parent", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    result = run(args.data, args.parent, args.arm, args.seed, args.seconds, args.threads,
                 args.output, args.learning_rate)
    print(json.dumps({key: result[key] for key in ("status", "arm", "seed", "elapsed_s")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
