"""Integer cubic-lattice construction and independent fold checks.

All public construction functions use the same origin/+x anchor and cube of
radius floor(length / 2) as the frozen CPU benchmark. No Torch dependency.
"""
from __future__ import annotations

import itertools
import random
import time

MOVES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_MOVE_INDEX = {v: i for i, v in enumerate(MOVES)}
_SYMMETRIES = tuple(itertools.product(itertools.permutations(range(3)),
                                      itertools.product((-1, 1), repeat=3)))


def check_sequence(sequence):
    if not isinstance(sequence, str) or len(sequence) < 2 or set(sequence) - {"H", "P"}:
        raise ValueError("Expected an HP sequence with at least two residues")


def validate_partial(sequence, positions, radius=None):
    check_sequence(sequence)
    radius = len(sequence) // 2 if radius is None else radius
    if not 2 <= len(positions) <= len(sequence):
        raise ValueError("Expected at least the two anchored residues")
    points = [tuple(p) for p in positions]
    if any(len(p) != 3 or any(type(v) is not int for v in p) for p in points):
        raise ValueError("Coordinates must be integer triples")
    if points[:2] != [(0, 0, 0), (1, 0, 0)]:
        raise ValueError("Incorrect origin/+x anchoring")
    if len(set(points)) != len(points):
        raise ValueError("Self intersection")
    if any(sum(abs(a - b) for a, b in zip(p, q)) != 1
           for p, q in zip(points, points[1:])):
        raise ValueError("Broken backbone")
    if any(abs(v) > radius for p in points for v in p):
        raise ValueError("Outside declared cube")
    return points


def contacts(sequence, positions, radius=None):
    """Count nonconsecutive HH neighbours once, requiring a complete witness."""
    points = validate_partial(sequence, positions, radius)
    if len(points) != len(sequence):
        raise ValueError("Incomplete folds are not scored")
    return sum(sequence[i] == sequence[j] == "H" and
               sum(abs(a - b) for a, b in zip(points[i], points[j])) == 1
               for i in range(len(sequence)) for j in range(i + 2, len(sequence)))


def validate(sequence, positions):
    return contacts(sequence, positions)


def legal_moves(sequence, positions):
    if len(positions) >= len(sequence):
        return []
    occupied, radius = {tuple(p) for p in positions}, len(sequence) // 2
    tail = positions[-1]
    return [a for a, move in enumerate(MOVES)
            if (point := tuple(x + d for x, d in zip(tail, move))) not in occupied
            and all(abs(v) <= radius for v in point)]


def incremental_contacts(sequence, positions, point):
    """New contacts when the next ordered residue is placed at point."""
    next_index = len(positions)
    if sequence[next_index] != "H":
        return 0
    return sum(sequence[i] == "H" and
               sum(abs(a - b) for a, b in zip(p, point)) == 1
               for i, p in enumerate(positions[:-1]))


def trajectory(sequence, positions):
    points = validate_partial(sequence, positions)
    actions = [_MOVE_INDEX[tuple(b - a for a, b in zip(p, q))]
               for p, q in zip(points, points[1:])]
    rewards = [0.0] + [float(incremental_contacts(sequence, points[:i], points[i]))
                       for i in range(2, len(points))]
    complete = len(points) == len(sequence)
    return {"sequence": sequence, "positions": [list(p) for p in points],
            "actions": actions, "rewards": rewards, "complete": complete,
            "contacts": validate(sequence, points) if complete else None,
            "trapped": not complete and not legal_moves(sequence, points),
            "interrupted": False}


def greedy(sequence, rng=None, attempts=100, deadline=None):
    """Random-tie maximum-immediate-contact construction, with trap restarts.

    Return the first complete fold, or None if all attempts fail/the monotonic
    deadline expires. A failed partial chain is never returned as a solution.
    """
    check_sequence(sequence)
    rng = random.Random() if rng is None else rng
    for _ in range(attempts):
        points = [(0, 0, 0), (1, 0, 0)]
        while len(points) < len(sequence):
            if deadline is not None and time.monotonic() >= deadline:
                return None
            legal = legal_moves(sequence, points)
            if not legal:
                break
            candidates = [tuple(x + d for x, d in zip(points[-1], MOVES[a])) for a in legal]
            gains = [incremental_contacts(sequence, points, p) for p in candidates]
            best = max(gains)
            points.append(rng.choice([p for p, gain in zip(candidates, gains) if gain == best]))
        if len(points) == len(sequence):
            return trajectory(sequence, points)
    return None


def _transform(positions, permutation, signs):
    return [tuple(signs[i] * p[permutation[i]] for i in range(3)) for p in positions]


def canonical_key(positions):
    """Deduplication key over all 48 signed permutations, preserving chain order."""
    points = [tuple(p) for p in positions]
    origin = points[0]
    points = [tuple(x - o for x, o in zip(p, origin)) for p in points]
    return min(tuple(_transform(points, permutation, signs))
               for permutation, signs in _SYMMETRIES)


def augment(sequence, positions, rng=None):
    """Rotate/reflect then re-anchor +x, transforming moves with the coordinates.

    The 48 cube symmetries induce eight distinct transforms after fixing the
    first bond. Residue order and HP labels are never reversed.
    """
    validate_partial(sequence, positions)
    rng = random.Random() if rng is None else rng
    permutation, signs = rng.choice(_SYMMETRIES)
    points = _transform(positions, permutation, signs)
    bond = points[1]
    axis = next(i for i, v in enumerate(bond) if v)
    reorder = (axis,) + tuple(i for i in range(3) if i != axis)
    points = _transform(points, reorder, (bond[axis], 1, 1))
    return trajectory(sequence, points)
