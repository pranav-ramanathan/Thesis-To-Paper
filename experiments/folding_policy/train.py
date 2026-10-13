"""Bounded, training-only pilot for one variable-length folding policy.

GRPO here means the clipped policy objective with within-sequence, group-normalized
outcome advantages. Outcomes sum the dense HH contact increments. A terminal
correction gives every trapped fold the declared negative return. The pilot uses
KL beta zero, no value network, and does not retry failures away before learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

CONDITIONS = ("rl_scratch", "patternboost", "cp_bc", "patternboost_rl", "cp_rl")
DEFAULTS = {
    "batch_size": 8, "group_size": 8, "learning_rate": 0.0003,
    "pretrain_fraction": 0.5, "data_fraction": 0.25, "elite_size": 8,
    "cp_seconds": 2.0, "validation_fraction": 0.10,
    "validation_folds_per_sequence": 2, "validation_checkpoints": 3,
    "trap_penalty": -10.0, "grpo_clip": 0.2, "grpo_epochs": 1,
    "grpo_kl_beta": 0.0, "temperature": 1.0, "gradient_clip": 1.0,
    "refresh_every_updates": 4, "regrow_probability": 0.5,
    "greedy_completion_probability": 0.5,
}


def sequence_key(sequence):
    """Reversal is also forbidden across train, validation and final test splits."""
    return min(sequence, sequence[::-1])


def check_splits(data):
    seen = {}
    for split in ("train", "validation", "test"):
        rows = data.get(split, [])
        if split != "test" and not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            seq = row["sequence"]
            if len(seq) < 3 or set(seq) - {"H", "P"}:
                raise ValueError(f"Invalid HP sequence in {split}")
            key = sequence_key(seq)
            if key in seen:
                raise ValueError(f"Duplicate or reversed sequence across splits: {split}/{seen[key]}")
            seen[key] = split
    train_keys = {sequence_key(row["sequence"]) for row in data["train"]}
    validation_keys = {sequence_key(row["sequence"]) for row in data["validation"]}
    for row in data.get("benchmark", []):
        if sequence_key(row["sequence"]) in train_keys | validation_keys:
            raise ValueError("A final benchmark sequence leaked into training/validation")


def group_advantages(rewards):
    """Population-normalized group rewards; identical groups contribute zero."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    if variance <= 1e-12:
        return [0.0] * len(rewards)
    scale = math.sqrt(variance)
    return [(r - mean) / scale for r in rewards]


def stratified_rows(rows, rng):
    """Visit every length/composition stratum before its second example."""
    buckets = {}
    for row in rows:
        sequence = row["sequence"]
        buckets.setdefault((len(sequence), sequence.count("H")), []).append(row)
    keys = list(buckets)
    rng.shuffle(keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result = []
    for index in range(max(map(len, buckets.values()))):
        result.extend(buckets[key][index] for key in keys if index < len(buckets[key]))
    return result


def outcome(rollout, trap_penalty):
    if rollout.get("interrupted"):
        return None
    if rollout["complete"]:
        dense = sum(rollout["rewards"])
        if abs(dense - rollout["contacts"]) > 1e-6:
            raise ValueError("Dense contact increments do not sum to final contacts")
        return float(dense)
    # Equivalent to adding trap_penalty - accumulated_contacts at termination.
    return float(trap_penalty)


def greedy_complete(sequence, prefix, rng, geometry, deadline):
    """One ordered greedy suffix completion of a proposed model prefix."""
    points = [list(point) for point in prefix]
    geometry.validate_partial(sequence, points)
    interrupted = False
    while len(points) < len(sequence):
        if time.monotonic() >= deadline:
            interrupted = True
            break
        legal = geometry.legal_moves(sequence, points)
        if not legal:
            break
        candidates = [[v + d for v, d in zip(points[-1], geometry.MOVES[action])] for action in legal]
        gains = [geometry.incremental_contacts(sequence, points, point) for point in candidates]
        maximum = max(gains)
        points.append(rng.choice([point for point, gain in zip(candidates, gains) if gain == maximum]))
    result = geometry.trajectory(sequence, points)
    result["interrupted"] = interrupted
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ElitePool:
    """Quality-ranked, symmetry-deduplicated elites with structural tie diversity."""
    def __init__(self, size, geometry):
        self.size, self.geometry = size, geometry
        self.by_sequence = {}

    @staticmethod
    def distance(left, right):
        # Pairwise lattice distances distinguish shapes without choosing an orientation.
        def signature(points):
            return tuple(sum(abs(a - b) for a, b in zip(points[i], points[j]))
                         for i in range(len(points)) for j in range(i + 2, len(points)))
        a, b = signature(left["positions"]), signature(right["positions"])
        return sum(x != y for x, y in zip(a, b))

    def add(self, rollout):
        if not rollout or not rollout["complete"]:
            return False
        seq = rollout["sequence"]
        score = self.geometry.validate(seq, rollout["positions"])
        if score != rollout["contacts"]:
            raise ValueError("Invalid elite score")
        key = self.geometry.canonical_key(rollout["positions"])
        previous = self.by_sequence.get(seq, [])
        if any(self.geometry.canonical_key(row["positions"]) == key for row in previous):
            return False
        candidates = previous + [rollout]
        selected = []
        while candidates and len(selected) < self.size:
            best_score = max(row["contacts"] for row in candidates)
            tied = [row for row in candidates if row["contacts"] == best_score]
            chosen = max(tied, key=lambda row: min(
                (self.distance(row, old) for old in selected), default=0))
            selected.append(chosen)
            candidates.remove(chosen)
        self.by_sequence[seq] = selected
        return any(row is rollout for row in selected)

    def __len__(self):
        return sum(len(rows) for rows in self.by_sequence.values())

    def draw(self, rng, count):
        sequences = list(self.by_sequence)
        if not sequences:
            return []
        return [rng.choice(self.by_sequence[rng.choice(sequences)]) for _ in range(count)]


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train(data_path, condition, seed, seconds, threads, output, config=None):
    started = time.monotonic()
    if condition not in CONDITIONS or seconds <= 0 or threads < 1:
        raise ValueError("Invalid condition, time budget or CPU thread count")
    import torch
    try:
        from . import geometry
        from .model import Policy, sample, supervised_loss, trajectory_log_probs
    except ImportError:
        import geometry
        from model import Policy, sample, supervised_loss, trajectory_log_probs
    config = config or {}
    settings = {**DEFAULTS, **config.get("training", {})}
    if settings["grpo_kl_beta"] != 0:
        raise ValueError("This declared GRPO-style pilot supports KL beta zero only")
    if not 0 < settings["pretrain_fraction"] < 1 or not 0 < settings["validation_fraction"] < 0.5:
        raise ValueError("Invalid phase budget fractions")
    if not 0 < settings["data_fraction"] < 1 or settings["trap_penalty"] >= 0:
        raise ValueError("Data fraction must be in (0,1); trap return must be negative")
    if settings["group_size"] < 2 or settings["validation_checkpoints"] < 1:
        raise ValueError("Need at least two GRPO samples and one validation checkpoint")
    if any(settings[name] <= 0 for name in ("batch_size", "elite_size", "cp_seconds", "temperature", "refresh_every_updates", "validation_folds_per_sequence", "grpo_epochs")):
        raise ValueError("Training sizes, temperature and durations must be positive")
    if any(not 0 <= settings[name] <= 1 for name in ("regrow_probability", "greedy_completion_probability")):
        raise ValueError("Regrowth probabilities must be between zero and one")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("result.json", "checkpoint.pt", "events.jsonl")):
        raise ValueError("Output already contains a run; choose a fresh directory")
    raw_data = Path(data_path).read_bytes()
    data = json.loads(raw_data)
    check_splits(data)
    if "split_sha256" in data:
        try:
            from .dataset import validate_dataset
        except ImportError:
            from dataset import validate_dataset
        validate_dataset(data)
    torch.set_num_threads(threads)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    policy = Policy(config.get("model", {})).cpu()
    if float(policy.config.get("dropout", 0)) != 0:
        raise ValueError("Pilot requires dropout zero for consistent masked GRPO ratios")
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings["learning_rate"])
    deadline = started + seconds
    optimize_deadline = deadline - seconds * settings["validation_fraction"]
    stop = {"requested": False}
    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda *_: stop.__setitem__("requested", True))
    events = (output / "events.jsonl").open("a", buffering=1)
    counters = {"generated_folds": 0, "valid_folds": 0, "interrupted_folds": 0,
                "supervised_examples": 0, "supervised_updates": 0, "rl_updates": 0,
                "rl_groups": 0, "zero_advantage_groups": 0, "cp_calls": 0,
                "cp_witness_records": 0, "retained_candidates": 0,
                "greedy_completions": 0,
                "validation_generated_folds": 0, "validation_valid_folds": 0}
    timing = {"initialization_s": time.monotonic() - started, "data_generation_s": 0.0,
              "supervised_s": 0.0, "candidate_generation_s": 0.0,
              "rl_s": 0.0, "validation_s": 0.0}
    pool = ElitePool(settings["elite_size"], geometry)
    data_coverage = {row["sequence"]: {"attempts": 0, "valid_records": 0} for row in data["train"]}
    snapshots, handoff = [], None
    midpoint_saved = False
    stage = "initialization"

    def emit(kind, **fields):
        events.write(json.dumps({"type": kind, "elapsed_s": time.monotonic() - started,
                                 "stage": stage, **fields}, sort_keys=True) + "\n")

    def active(limit):
        return not stop["requested"] and time.monotonic() < min(limit, optimize_deadline)

    def observe(rollouts):
        counters["generated_folds"] += len(rollouts)
        counters["valid_folds"] += sum(bool(row["complete"]) for row in rollouts)
        counters["interrupted_folds"] += sum(bool(row.get("interrupted")) for row in rollouts)

    def coverage():
        return [{"id": row.get("id", f"train_{index}"),
                 "length": len(row["sequence"]),
                 "hydrophobic_fraction": row["sequence"].count("H") / len(row["sequence"]),
                 "initial_data_attempts": data_coverage[row["sequence"]]["attempts"],
                 "initial_data_valid_records": data_coverage[row["sequence"]]["valid_records"],
                 "elite_folds": len(pool.by_sequence.get(row["sequence"], []))}
                for index, row in enumerate(data["train"])]

    def save_pool():
        identifiers = {row["sequence"]: row.get("id", f"train_{index}") for index, row in enumerate(data["train"])}
        atomic_json(output / "elite_pool.json", {
            "schema": 1, "condition": condition,
            "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
            "folds": [{"sequence_id": identifiers[seq], "sequence": seq,
                       "positions": row["positions"], "contacts": row["contacts"],
                       "source": row.get("source")}
                      for seq, rows in pool.by_sequence.items() for row in rows]})

    def snapshot(label):
        state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        snapshots.append({"label": label, "state": state, "elapsed_s": time.monotonic() - started,
                          "updates": counters["supervised_updates"] + counters["rl_updates"]})
        # Final and handoff states stay affordable: at most the declared number are evaluated.
        if len(snapshots) > settings["validation_checkpoints"]:
            del snapshots[0]

    def maybe_midpoint():
        nonlocal midpoint_saved
        if condition in ("rl_scratch", "patternboost", "cp_bc") and not midpoint_saved and time.monotonic() >= started + (optimize_deadline - started) * 0.5:
            snapshot(stage + "_midpoint")
            midpoint_saved = True

    def supervised_step():
        rows = pool.draw(rng, settings["batch_size"])
        if not rows:
            return False
        augmented = [geometry.augment(row["sequence"], row["positions"], rng) for row in rows]
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(policy, augmented)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite supervised loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["gradient_clip"])
        optimizer.step()
        counters["supervised_updates"] += 1
        counters["supervised_examples"] += len(rows)
        emit("supervised_update", loss=float(loss.detach()), examples=len(rows), elite_count=len(pool))
        return True

    def initial_greedy(limit):
        begin = time.monotonic()
        index = 0
        schedule = stratified_rows(data["train"], rng)
        while active(limit):
            seq = schedule[index % len(schedule)]["sequence"]
            data_coverage[seq]["attempts"] += 1
            # One attempt at a time keeps failed construction counts observable.
            row = geometry.greedy(seq, rng=rng, attempts=1, deadline=min(limit, optimize_deadline))
            counters["generated_folds"] += 1
            if row is not None and row["complete"]:
                row["source"] = "randomized_greedy"
                counters["valid_folds"] += 1
                data_coverage[seq]["valid_records"] += 1
                counters["retained_candidates"] += int(pool.add(row))
            elif time.monotonic() >= min(limit, optimize_deadline):
                counters["interrupted_folds"] += 1
            index += 1
            if index >= len(data["train"]) and len(pool) >= len(data["train"]) * settings["elite_size"]:
                break
        timing["data_generation_s"] += time.monotonic() - begin
        emit("greedy_data", attempts=index, elite_count=len(pool))

    def cp_data(limit):
        begin = time.monotonic()
        cluster = Path(__file__).resolve().parents[1] / "cluster"
        # hp_solver imports a top-level geometry; load both in an isolated temporary binding.
        import importlib.util
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        old_geometry = sys.modules.get("geometry")
        sys.modules["geometry"] = load("_folding_cp_geometry", cluster / "geometry.py")
        try:
            solver = load("_folding_cp_solver", cluster / "hp_solver.py")
        finally:
            if old_geometry is None:
                sys.modules.pop("geometry", None)
            else:
                sys.modules["geometry"] = old_geometry
        index = 0
        schedule = stratified_rows(data["train"], rng)
        while active(limit):
            seq = schedule[index % len(schedule)]["sequence"]
            remaining = min(limit, optimize_deadline) - time.monotonic()
            if remaining <= 0:
                break
            def receive(event):
                if event["type"] == "witness":
                    data_coverage[seq]["valid_records"] += 1
                    counters["cp_witness_records"] += 1
                    counters["generated_folds"] += 1
                    counters["valid_folds"] += 1
                    if time.monotonic() <= min(limit, optimize_deadline):
                        row = geometry.trajectory(seq, event["positions"])
                        row["source"] = "cp_sat"
                        counters["retained_candidates"] += int(pool.add(row))
            data_coverage[seq]["attempts"] += 1
            solver.solve(seq, seconds=min(settings["cp_seconds"], remaining), threads=threads,
                         seed=rng.randrange(2**31 - 1), emit=receive,
                         stopped=lambda: not active(limit))
            counters["cp_calls"] += 1
            index += 1
        timing["data_generation_s"] += time.monotonic() - begin
        emit("cp_data", calls=index, witnesses=counters["cp_witness_records"], elite_count=len(pool))

    def refresh(limit):
        begin = time.monotonic()
        sequences = [rng.choice(data["train"])["sequence"] for _ in range(settings["batch_size"])]
        prefixes = []
        for seq in sequences:
            elites = pool.by_sequence.get(seq, [])
            if elites and rng.random() < settings["regrow_probability"]:
                fold = rng.choice(elites)["positions"]
                prefixes.append(fold[:rng.randrange(2, len(fold))])
            else:
                prefixes.append(None)
        policy.eval()
        rows = sample(policy, sequences, generator=generator, temperature=settings["temperature"],
                      prefixes=prefixes, deadline=min(limit, optimize_deadline))
        for row in rows:
            row["source"] = "model_or_model_suffix_regrowth"
        observe(rows)
        retained = sum(int(pool.add(row)) for row in rows if not row.get("interrupted"))
        # Training-only local completion: keep the model sample as well as a
        # greedily completed prefix. Evaluation never imports this operation.
        completions = []
        for row in rows:
            if not active(limit):
                break
            if len(row["positions"]) > 2 and rng.random() < settings["greedy_completion_probability"]:
                prefix = row["positions"][:rng.randrange(2, len(row["positions"]))]
                completions.append(greedy_complete(row["sequence"], prefix, rng, geometry,
                                                   min(limit, optimize_deadline)))
                completions[-1]["source"] = "model_prefix_greedy_completion"
        observe(completions)
        counters["greedy_completions"] += len(completions)
        retained += sum(int(pool.add(row)) for row in completions if not row.get("interrupted"))
        counters["retained_candidates"] += retained
        timing["candidate_generation_s"] += time.monotonic() - begin
        emit("pool_refresh", generated=len(rows), greedy_completions=len(completions),
             retained=retained, elite_count=len(pool))

    def pretrain(limit, source):
        nonlocal stage
        stage = "patternboost" if source == "greedy" else "cp_bc"
        allocation_start = time.monotonic()
        data_limit = allocation_start + max(0, limit - allocation_start) * settings["data_fraction"]
        (initial_greedy if source == "greedy" else cp_data)(data_limit)
        if not len(pool):
            raise RuntimeError("No valid training examples within the allocated data budget")
        while active(limit):
            begin = time.monotonic()
            supervised_step()
            timing["supervised_s"] += time.monotonic() - begin
            if source == "greedy" and counters["supervised_updates"] % settings["refresh_every_updates"] == 0 and active(limit):
                refresh(limit)
            maybe_midpoint()
        snapshot(stage)
        receipt = {"state_sha256": state_hash(policy.state_dict()), "model_object": id(policy),
                   "optimizer_object": id(optimizer), "supervised_updates": counters["supervised_updates"]}
        emit("pretraining_end", state_sha256=receipt["state_sha256"],
             supervised_updates=receipt["supervised_updates"])
        return receipt

    def reinforcement(limit, pretraining_receipt=None):
        nonlocal stage, handoff
        stage = "rl"
        if pretraining_receipt is not None:
            handoff = {"pretraining_sha256": pretraining_receipt["state_sha256"],
                       "rl_initial_sha256": state_hash(policy.state_dict()),
                       "same_model_object": id(policy) == pretraining_receipt["model_object"],
                       "optimizer_state_retained": id(optimizer) == pretraining_receipt["optimizer_object"],
                       "supervised_updates": pretraining_receipt["supervised_updates"]}
            if handoff["pretraining_sha256"] != handoff["rl_initial_sha256"] or not handoff["same_model_object"]:
                raise RuntimeError("Pretraining weights were not handed unchanged to RL")
            emit("stage_handoff", **handoff)
        begin = time.monotonic()
        while active(limit):
            maybe_midpoint()
            seq = rng.choice(data["train"])["sequence"]
            policy.eval()
            rows = sample(policy, [seq] * settings["group_size"], generator=generator,
                          temperature=settings["temperature"], deadline=min(limit, optimize_deadline))
            observe(rows)
            rows = [row for row in rows if not row.get("interrupted")]
            if len(rows) < 2 or not active(limit):
                break
            rewards = [outcome(row, settings["trap_penalty"]) for row in rows]
            advantages = group_advantages(rewards)
            counters["rl_groups"] += 1
            if not any(advantages):
                counters["zero_advantage_groups"] += 1
                emit("rl_group", rewards=rewards, zero_advantage=True, updated=False)
                continue
            with torch.no_grad():
                old, taken = trajectory_log_probs(policy, rows, temperature=settings["temperature"])
            if not bool(taken.any()):
                continue
            advantage_tensor = torch.tensor(advantages, dtype=old.dtype, device=old.device).unsqueeze(1)
            loss_value, updated = None, False
            for _ in range(settings["grpo_epochs"]):
                if not active(limit):
                    break
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                current, current_taken = trajectory_log_probs(policy, rows, temperature=settings["temperature"])
                if not torch.equal(taken, current_taken):
                    raise RuntimeError("GRPO legal-action masks changed on the same trajectories")
                ratio = torch.exp(current - old)
                clipped = ratio.clamp(1 - settings["grpo_clip"], 1 + settings["grpo_clip"])
                surrogate = torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor)
                loss = -((surrogate * taken).sum(1) / taken.sum(1).clamp_min(1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite GRPO loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), settings["gradient_clip"])
                optimizer.step()
                counters["rl_updates"] += 1
                loss_value, updated = float(loss.detach()), True
            emit("rl_group", rewards=rewards, zero_advantage=False, updated=updated,
                 valid=sum(row["complete"] for row in rows), loss=loss_value)
        timing["rl_s"] += time.monotonic() - begin
        snapshot(stage)

    def checkpoint(state, label, selection):
        return {"schema": 1, "model_config": policy.config, "model_state": state,
                "condition": condition, "seed": seed, "stage": label,
                "selection": selection, "state_sha256": state_hash(state),
                "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
                "config": {"model": policy.config, "training": settings},
                "optimizer_retained_across_stages": True}

    def save_checkpoint(name, value):
        destination = output / name
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, destination)

    status, error = "complete", None
    try:
        emit("start", condition=condition, seed=seed, seconds=seconds, threads=threads,
             dataset_sha256=hashlib.sha256(raw_data).hexdigest(),
             config={"model": policy.config, "training": settings})
        initial_hash = state_hash(policy.state_dict())
        if condition == "rl_scratch":
            reinforcement(optimize_deadline)
        elif condition in ("patternboost", "cp_bc"):
            pretrain(optimize_deadline, "greedy" if condition == "patternboost" else "cp")
        else:
            limit = started + (optimize_deadline - started) * settings["pretrain_fraction"]
            receipt = pretrain(limit, "greedy" if condition == "patternboost_rl" else "cp")
            reinforcement(optimize_deadline, receipt)
        final_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        # Validation uses its own fixed random stream; training random state is untouched.
        stage = "validation"
        best = None
        begin = time.monotonic()
        candidates = snapshots[-settings["validation_checkpoints"]:]
        requested = len(data["validation"]) * settings["validation_folds_per_sequence"]
        for index, candidate in enumerate(candidates):
            if stop["requested"] or time.monotonic() >= deadline:
                break
            policy.load_state_dict(candidate["state"])
            policy.eval()
            validation_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
            sequences = [row["sequence"] for row in data["validation"]
                         for _ in range(settings["validation_folds_per_sequence"])]
            candidate_deadline = time.monotonic() + max(0, deadline - time.monotonic()) / (len(candidates) - index)
            rows = sample(policy, sequences, generator=validation_generator,
                          temperature=settings["temperature"], deadline=candidate_deadline)
            counters["validation_generated_folds"] += len(rows)
            counters["validation_valid_folds"] += sum(row["complete"] for row in rows)
            covered = len(rows) == requested and not any(row.get("interrupted") for row in rows)
            selection = {"split": "validation", "validated": covered, "samples": len(rows),
                         "requested_samples": requested,
                         "mean_contacts": sum(row["contacts"] if row["complete"] else 0 for row in rows) / max(1, len(rows)),
                         "valid_rate": sum(row["complete"] for row in rows) / max(1, len(rows)),
                         "incomplete_selection_score": 0,
                         "candidate_training_elapsed_s": candidate["elapsed_s"]}
            emit("validation", label=candidate["label"], **selection)
            if covered and (best is None or (selection["mean_contacts"], selection["valid_rate"]) >
                            (best["selection"]["mean_contacts"], best["selection"]["valid_rate"])):
                best = {"state": candidate["state"], "label": candidate["label"], "selection": selection}
        timing["validation_s"] += time.monotonic() - begin
        if best is None:
            status = "validation_incomplete"
            best = {"state": final_state, "label": "final_unvalidated",
                    "selection": {"split": "validation", "validated": False, "samples": 0}}
        if stop["requested"]:
            status = "interrupted"
        elif condition in ("rl_scratch", "patternboost_rl", "cp_rl") and counters["rl_updates"] == 0:
            status = "insufficient_training"
        elif condition != "rl_scratch" and counters["supervised_updates"] == 0:
            status = "insufficient_training"
        computation_elapsed = time.monotonic() - started
        save_checkpoint("last.pt", checkpoint(final_state, "final", {"split": "none", "validated": False}))
        save_checkpoint("checkpoint.pt", checkpoint(best["state"], best["label"], best["selection"]))
        save_pool()
        result = {"schema": 1, "status": status, "condition": condition, "seed": seed,
                  "requested_seconds": seconds, "compute_elapsed_s": computation_elapsed,
                  "elapsed_s": time.monotonic() - started, "threads": threads,
                  "model_config": policy.config, "training_settings": settings,
                  "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
                  "initial_state_sha256": initial_hash, "final_state_sha256": state_hash(final_state),
                  "selected_state_sha256": state_hash(best["state"]), "handoff": handoff,
                  "selection": best["selection"], "counters": counters, "timing": timing,
                  "elite_count": len(pool), "training_sequences": len(data["train"]),
                  "elite_sequence_count": len(pool.by_sequence),
                  "training_example_coverage": coverage(),
                  "validation_sequences": len(data["validation"]), "test_evaluated": False,
                  "valid_fold_rate": counters["valid_folds"] / max(1, counters["generated_folds"]),
                  "zero_advantage_group_fraction": counters["zero_advantage_groups"] / max(1, counters["rl_groups"]),
                  "limitations": ["GRPO-style clipped outcome objective; KL beta zero.",
                                  "Budgets include data generation, training and validation; a running tensor operation can overrun the deadline.",
                                  "Checkpoint serialization occurs after computation and is timed separately by elapsed_s.",
                                  "Validation failures have selection score zero; test inference never accepts incomplete folds."]}
        atomic_json(output / "result.json", result)
        emit("finished", status=status, counters=counters)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        save_pool()
        atomic_json(output / "result.json", {"schema": 1, "status": "failed", "error": error,
                    "condition": condition, "seed": seed, "elapsed_s": time.monotonic() - started,
                    "requested_seconds": seconds, "threads": threads,
                    "model_config": policy.config, "training_settings": settings,
                    "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
                    "counters": counters, "timing": timing, "test_evaluated": False,
                    "elite_count": len(pool), "elite_sequence_count": len(pool.by_sequence),
                    "training_sequences": len(data["train"]), "training_example_coverage": coverage()})
        emit("failed", error=error)
        raise
    finally:
        events.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    result = train(args.data, args.condition, args.seed, args.seconds, args.threads, args.output, config)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()),
                      "condition": args.condition, "counters": result["counters"]}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
