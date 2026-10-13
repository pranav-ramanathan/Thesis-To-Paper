"""One small variable-length policy for cloning, elite training, and GRPO.

The full HP sequence is visible at every action. Coordinates are visible only
for the placed prefix: bidirectional attention over sequence tokens therefore
does not reveal future geometry. RoPE has no learned length-specific table.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .geometry import (MOVES, check_sequence, legal_moves, trajectory,
                           validate_partial)
except ImportError:
    from geometry import (MOVES, check_sequence, legal_moves, trajectory,
                          validate_partial)


def _rope(x):
    length, dim = x.shape[-2:]
    frequencies = torch.exp(-math.log(10000.0) *
                            torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * frequencies
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class _Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, padding_mask):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(_rope(q), _rope(k), v,
                                                  attn_mask=~padding_mask[:, None, None, :],
                                                  dropout_p=0.0)
        x = x + self.project(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.ff(self.norm2(x))


class Policy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        supplied = {} if config is None else dict(config)
        self.config = {"width": int(supplied.get("width", 64)),
                       "heads": int(supplied.get("heads", 4)),
                       "layers": int(supplied.get("layers", 2)), "dropout": 0.0}
        if supplied.get("dropout", 0.0) != 0:
            raise ValueError("This policy requires dropout=0 for consistent GRPO likelihood ratios")
        width, heads, layers = (self.config[k] for k in ("width", "heads", "layers"))
        if width <= 0 or heads <= 0 or layers <= 0 or width % heads or (width // heads) % 2:
            raise ValueError("width/heads must be a positive even integer for RoPE")
        self.residue = nn.Embedding(3, width, padding_idx=2)
        self.placed = nn.Embedding(2, width)
        self.coordinates = nn.Linear(3, width, bias=False)
        self.blocks = nn.ModuleList([_Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 6)

    def forward(self, residues, coordinates, placed, padding_mask):
        """Return six move logits at the next unplaced residue of each prefix.

        residues: H=0/P=1/padding=2, coordinates: B,L,3; both masks B,L.
        padding_mask=True means padding. Unplaced coordinate values are ignored.
        """
        visible = placed & ~padding_mask
        radius = ((~padding_mask).sum(1) // 2).clamp_min(1).to(coordinates.dtype)
        xyz = coordinates * visible.unsqueeze(-1) / radius[:, None, None]
        x = self.residue(residues) + self.placed(visible.long()) + self.coordinates(xyz)
        for block in self.blocks:
            x = block(x, padding_mask)
        query = visible.sum(1).clamp_max(x.shape[1] - 1)
        state = x[torch.arange(x.shape[0], device=x.device), query]
        return self.output(self.norm(state))


def encode_states(sequences, prefixes, device):
    """Pack variable-length sequences; geometry after the prefix is always zero."""
    length = max(map(len, sequences))
    residues = torch.full((len(sequences), length), 2, dtype=torch.long, device=device)
    xyz = torch.zeros((len(sequences), length, 3), dtype=torch.float32, device=device)
    placed = torch.zeros((len(sequences), length), dtype=torch.bool, device=device)
    padding = torch.ones_like(placed)
    for i, (sequence, prefix) in enumerate(zip(sequences, prefixes)):
        residues[i, :len(sequence)] = torch.tensor([int(c == "P") for c in sequence], device=device)
        xyz[i, :len(prefix)] = torch.tensor(prefix, dtype=torch.float32, device=device)
        placed[i, :len(prefix)] = True
        padding[i, :len(sequence)] = False
    return residues, xyz, placed, padding


def _legal_tensor(sequences, prefixes, device):
    mask = torch.zeros((len(sequences), 6), dtype=torch.bool, device=device)
    for i, (sequence, prefix) in enumerate(zip(sequences, prefixes)):
        mask[i, legal_moves(sequence, prefix)] = True
    return mask


def _temperature(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and positive")


def sample(policy, sequences, generator=None, temperature=1.0, prefixes=None, deadline=None):
    """Batched standalone rollouts. No solver calls, repairs, or implicit retries.

    A retry is a new counted rollout chosen by the caller. A monotonic deadline
    is checked before and after every neural forward pass. Trap/deadline partial
    folds have contacts=None and never count as completed solutions.
    """
    _temperature(temperature)
    sequences = list(sequences)
    if not sequences:
        return []
    if prefixes is not None and len(prefixes) != len(sequences):
        raise ValueError("One optional prefix is required per sequence")
    points = []
    for i, sequence in enumerate(sequences):
        check_sequence(sequence)
        prefix = None if prefixes is None else prefixes[i]
        prefix = [(0, 0, 0), (1, 0, 0)] if prefix is None else prefix
        points.append([tuple(p) for p in validate_partial(sequence, prefix)])
    device = next(policy.parameters()).device
    was_training = policy.training
    interrupted = [False] * len(sequences)
    policy.eval()
    try:
        with torch.inference_mode():
            while True:
                active = [i for i, seq in enumerate(sequences)
                          if len(points[i]) < len(seq) and legal_moves(seq, points[i])]
                if not active:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    for i in active:
                        interrupted[i] = True
                    break
                seqs, states = [sequences[i] for i in active], [points[i] for i in active]
                logits = policy(*encode_states(seqs, states, device)) / temperature
                legal = _legal_tensor(seqs, states, device)
                probs = logits.masked_fill(~legal, -torch.inf).softmax(-1)
                if deadline is not None and time.monotonic() >= deadline:
                    for i in active:
                        interrupted[i] = True
                    break
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(1).tolist()
                for i, action in zip(active, actions):
                    points[i].append(tuple(v + d for v, d in zip(points[i][-1], MOVES[action])))
    finally:
        policy.train(was_training)
    results = [trajectory(seq, p) for seq, p in zip(sequences, points)]
    for result, stopped in zip(results, interrupted):
        result["interrupted"] = stopped
    return results


def trajectory_log_probs(policy, trajectories, temperature=1.0, chunk_size=64):
    """Differentiable log probabilities and token mask, excluding fixed +x.

    Uses the identical legal-move mask and temperature as sampling. Each state
    is reconstructed from its strict prefix, never the completed coordinates.
    Padding and missing actions in trapped rollouts are excluded from the mask.
    """
    _temperature(temperature)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    device = next(policy.parameters()).device
    width = max((len(t["positions"]) - 2 for t in trajectories), default=0)
    batch = len(trajectories)
    states, sequences, selected, flat_indices = [], [], [], []
    for i, item in enumerate(trajectories):
        sequence, points = item["sequence"], item["positions"]
        validate_partial(sequence, points)
        actions = trajectory(sequence, points)["actions"]
        for j in range(2, len(points)):
            sequences.append(sequence)
            states.append(points[:j])
            selected.append(actions[j - 1])
            flat_indices.append(i * width + j - 2)
    # Retain a zero gradient path for groups with no sampled (non-anchor) actions.
    base = next(policy.parameters()).sum() * 0 + torch.zeros(batch * width, device=device)
    mask = torch.zeros(batch * width, dtype=torch.bool, device=device)
    values = []
    for start in range(0, len(states), chunk_size):
        end = start + chunk_size
        seqs, prefixes = sequences[start:end], states[start:end]
        logits = policy(*encode_states(seqs, prefixes, device)) / temperature
        legal = _legal_tensor(seqs, prefixes, device)
        chosen = torch.tensor(selected[start:end], dtype=torch.long, device=device)
        logp = logits.masked_fill(~legal, -torch.inf).log_softmax(-1)
        values.append(logp.gather(1, chosen[:, None]).squeeze(1))
    if values:
        indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
        base = base.scatter(0, indices, torch.cat(values))
        mask[indices] = True
    return base.reshape(batch, width), mask.reshape(batch, width)


def supervised_loss(policy, trajectories):
    logp, mask = trajectory_log_probs(policy, trajectories)
    return -(logp * mask).sum() / mask.sum().clamp_min(1)


# Concise alias for callers that only need the per-action likelihood interface.
log_probs = trajectory_log_probs
