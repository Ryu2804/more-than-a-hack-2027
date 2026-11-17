"""Refine baseline boundaries without changing each half's ordered run labels."""

import numpy as np
from numba import njit

import bridge as b


CONFIGS = [
    dict(name=f"radius{radius}_penalty{penalty:g}", radius=radius, penalty=penalty)
    for radius in (2, 5, 10)
    for penalty in (0.0, 0.05)
]


@njit(cache=False)
def _optimal_boundaries(prefix, original, labels, radius, penalty):
    count = len(original)
    length = len(prefix) - 1
    width = 2 * radius + 1
    lower = np.empty(count, dtype=np.int64)
    upper = np.empty(count, dtype=np.int64)
    next_index = np.full((count, width), -1, dtype=np.int32)
    for j in range(count):
        lower[j] = max(j + 1, original[j] - radius)
        upper[j] = min(length - (count - j), original[j] + radius)

    # Fixed-label segment sums telescope into one unary gain per boundary.
    scores = np.full(width, -np.inf)
    current = np.empty(width, dtype=np.float64)
    last = count - 1
    for k in range(upper[last] - lower[last] + 1):
        position = lower[last] + k
        displacement = (position - original[last]) / radius
        scores[k] = (prefix[position, labels[last]]
                     - prefix[position, labels[last + 1]]
                     - penalty * displacement * displacement)

    # Suffix DP resolves ties at the earliest next boundary, then traces forward.
    for j in range(count - 2, -1, -1):
        current.fill(-np.inf)
        candidate = upper[j + 1] - lower[j + 1]
        best_score = -np.inf
        best_index = -1
        for k in range(upper[j] - lower[j], -1, -1):
            position = lower[j] + k
            while candidate >= 0 and lower[j + 1] + candidate > position:
                value = scores[candidate]
                if value > best_score or (value == best_score and value > -np.inf):
                    best_score = value
                    best_index = candidate
                candidate -= 1
            if best_index >= 0:
                displacement = (position - original[j]) / radius
                current[k] = (prefix[position, labels[j]]
                              - prefix[position, labels[j + 1]]
                              - penalty * displacement * displacement
                              + best_score)
                next_index[j, k] = best_index
        scores, current = current, scores

    best_index = 0
    for k in range(1, upper[0] - lower[0] + 1):
        if scores[k] > scores[best_index]:
            best_index = k
    result = np.empty(count, dtype=np.int64)
    for j in range(count):
        result[j] = lower[j] + best_index
        if j + 1 < count:
            best_index = next_index[j, best_index]
    return result


def relocate(codes, probability, blocks, radius, penalty):
    """Maximize joint log probability minus squared normalized displacement.

    Half endpoints and ordered labels stay fixed; runs may shrink to one sample.
    Equal optima use lexicographically earliest internal boundary positions.
    Inputs are read-only, and the returned codes are a fresh int8 array.
    """
    codes = np.asarray(codes)
    probability = np.asarray(probability, dtype=np.float64)
    if codes.ndim != 1 or not np.isin(codes, (0, 1, 2)).all():
        raise ValueError("codes must be a vector containing only 0, 1, 2")
    if (probability.shape != (len(codes), 3)
            or not np.isfinite(probability).all() or np.any(probability < 0)):
        raise ValueError("probability must be a finite nonnegative (N, 3) array")
    if not np.isfinite(radius) or int(radius) != radius or radius <= 0:
        raise ValueError("radius must be a positive integer number of samples")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    radius = int(radius)
    penalty = float(penalty)
    blocks = b.decoder._normalize_blocks(blocks, len(codes))
    result = np.array(codes, dtype=np.int8, copy=True)
    for start, end in blocks:
        half = codes[start:end]
        original = np.flatnonzero(half[1:] != half[:-1]) + 1
        if not len(original):
            continue
        edges = np.r_[0, original, len(half)]
        labels = half[edges[:-1]].astype(np.int8)
        prefix = np.empty((len(half) + 1, 3), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(np.log(np.maximum(probability[start:end], 1e-8)), axis=0,
                  out=prefix[1:])
        moved = _optimal_boundaries(prefix, original, labels, radius, penalty)
        edges[1:-1] = moved
        for j, label in enumerate(labels):
            result[start + edges[j]:start + edges[j + 1]] = label
    return result


def decode(prediction, observations, stats, config, baseline_codes=None):
    """Use cached baseline codes when supplied; never fit on observations."""
    if baseline_codes is None:
        baseline_codes = b.baseline(prediction, observations, stats)
    return relocate(baseline_codes, prediction["probability"], observations["blocks"],
                    config["radius"], config["penalty"])
