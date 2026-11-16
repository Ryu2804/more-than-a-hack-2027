import math

import numpy as np
from numba import njit


CONFIGS = [
    {'name': 'cost05', 'lambda_team': 0.5, 'lambda_dead': 0.5, 'gamma': 0.0},
    {'name': 'cost2', 'lambda_team': 2.0, 'lambda_dead': 2.0, 'gamma': 0.0},
    {'name': 'cost6', 'lambda_team': 6.0, 'lambda_dead': 6.0, 'gamma': 0.0},
    {'name': 'team05_dead2', 'lambda_team': 0.5, 'lambda_dead': 2.0, 'gamma': 0.0},
    {'name': 'team2_dead05', 'lambda_team': 2.0, 'lambda_dead': 0.5, 'gamma': 0.0},
    {'name': 'centered2', 'lambda_team': 2.0, 'lambda_dead': 2.0, 'gamma': 0.5},
]

_EPSILON = 1e-6


@njit(cache=False)
def _logit(value):
    if value < _EPSILON:
        value = _EPSILON
    elif value > 1.0 - _EPSILON:
        value = 1.0 - _EPSILON
    return math.log(value) - math.log(1.0 - value)


@njit(cache=False)
def _decode_block(probability, boundary, start, end, lambda_team, lambda_dead, gamma, output):
    length = end - start
    if length == 1:
        state = 0
        if probability[start, 1] > probability[start, state]:
            state = 1
        if probability[start, 2] > probability[start, state]:
            state = 2
        output[start] = state
        return

    mean_logit = 0.0
    if gamma != 0.0:
        mean_boundary = 0.0
        for t in range(start, end):
            mean_boundary += boundary[t]
        mean_logit = _logit(mean_boundary / length)

    back = np.empty((length, 3), dtype=np.int8)
    score = np.empty(3, dtype=np.float64)
    next_score = np.empty(3, dtype=np.float64)
    for state in range(3):
        score[state] = math.log(probability[start, state])

    for local_t in range(1, length):
        t = start + local_t
        centered_boundary = 0.0
        if gamma != 0.0:
            centered_boundary = _logit(boundary[t]) - mean_logit

        for destination in range(3):
            best = score[destination]
            origin = destination
            for source in range(3):
                if source == destination:
                    continue
                if source < 2 and destination < 2:
                    cost = lambda_team
                else:
                    cost = lambda_dead
                cost -= gamma * centered_boundary
                if cost < 0.0:
                    cost = 0.0
                candidate = score[source] - cost
                if candidate > best:
                    best = candidate
                    origin = source
            next_score[destination] = best + math.log(probability[t, destination])
            back[local_t, destination] = origin

        score, next_score = next_score, score

    state = 0
    if score[1] > score[state]:
        state = 1
    if score[2] > score[state]:
        state = 2
    output[end - 1] = state
    for local_t in range(length - 1, 0, -1):
        state = back[local_t, state]
        output[start + local_t - 1] = state


def _validated_blocks(blocks, length):
    try:
        spans = np.asarray(blocks)
    except (TypeError, ValueError) as exc:
        raise ValueError('blocks must be integer [start, end) pairs') from exc

    if spans.ndim == 1 and spans.size == 0:
        spans = np.empty((0, 2), dtype=np.int64)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError('blocks must be integer [start, end) pairs')
    if not np.issubdtype(spans.dtype, np.integer) or np.issubdtype(spans.dtype, np.bool_):
        raise ValueError('block boundaries must be integers')
    if spans.shape[0] == 0:
        if length != 0:
            raise ValueError('blocks must partition the full sequence')
        return spans

    expected_start = 0
    for start, end in spans:
        if start != expected_start or end <= start or end > length:
            raise ValueError('blocks must partition the full sequence')
        expected_start = int(end)
    if expected_start != length:
        raise ValueError('blocks must partition the full sequence')
    return spans


def decode(probability, boundary, blocks, config):
    try:
        probability = np.asarray(probability, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('probability must be a finite Nx3 array') from exc
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError('probability must have shape (N, 3)')
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError('probabilities must be finite and nonnegative')
    mass = probability.sum(axis=1)
    if not np.all(np.isfinite(mass)) or not np.allclose(mass, 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError('probability rows must sum to one')
    probability = np.maximum(probability, 1e-8)
    probability /= probability.sum(axis=1, keepdims=True)

    length = probability.shape[0]
    try:
        boundary = np.asarray(boundary, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('boundary must be a finite length-N array in [0, 1]') from exc
    if boundary.shape != (length,):
        raise ValueError('boundary must have shape (N,)')
    if not np.all(np.isfinite(boundary)) or np.any(boundary < 0.0) or np.any(boundary > 1.0):
        raise ValueError('boundary values must be finite and in [0, 1]')

    spans = _validated_blocks(blocks, length)
    try:
        lambda_team = float(config['lambda_team'])
        lambda_dead = float(config['lambda_dead'])
        gamma = float(config['gamma'])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('config must define lambda_team, lambda_dead, and gamma') from exc
    if (not np.isfinite(lambda_team) or not np.isfinite(lambda_dead) or not np.isfinite(gamma)
            or lambda_team < 0.0 or lambda_dead < 0.0 or gamma < 0.0):
        raise ValueError('transition parameters must be finite and nonnegative')

    output = np.empty(length, dtype=np.int8)
    for start, end in spans:
        _decode_block(probability, boundary, int(start), int(end), lambda_team, lambda_dead, gamma, output)
    return output
