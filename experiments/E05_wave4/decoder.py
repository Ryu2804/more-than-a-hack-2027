from collections.abc import Mapping
import math

import numpy as np

from numba import njit

FPS = 5
LAMBDA_GRID = (0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 120.0)
_EPS = 1e-6

_DEFAULT_CONFIG = {
    "gate_r": 0.35,
    "kappa": 0.05,
    "merge_s": 1.0,
    "dead_threshold": 0.5,
    "min_dead_s": 2.0,
    "tau": 3.0,
    "rate_scale": 1.0,
    "shift": True,
    "lambda_mode": "rate",
    "fixed_lambda": 2.0,
}

CONFIGS = [
    {"name": "ryu_rate", **_DEFAULT_CONFIG},
    {"name": "gate10", **{**_DEFAULT_CONFIG, "gate_r": 1.0}},
    {"name": "merge05", **{**_DEFAULT_CONFIG, "merge_s": 0.5}},
    {"name": "kappa02", **{**_DEFAULT_CONFIG, "kappa": 0.02}},
    {"name": "no_shift", **{**_DEFAULT_CONFIG, "shift": False}},
    {"name": "fixed2", **{**_DEFAULT_CONFIG, "lambda_mode": "fixed"}},
    {"name": "rate08", **{**_DEFAULT_CONFIG, "rate_scale": 0.8}},
    {"name": "rate12", **{**_DEFAULT_CONFIG, "rate_scale": 1.2}},
]


@njit(cache=False)
def viterbi_2state(lp_h, lp_a, switch_cost):
    """Maximize team log-evidence minus switch costs; ties prefer H."""
    n = len(lp_h)
    path = np.empty(n, dtype=np.int8)
    if n == 0:
        return path

    back = np.empty((n, 2), dtype=np.int8)
    score_h = lp_h[0]
    score_a = lp_a[0]
    for t in range(1, n):
        from_h_to_h = score_h
        from_a_to_h = score_a - switch_cost
        if from_h_to_h >= from_a_to_h:
            next_h = from_h_to_h
            back[t, 0] = 0
        else:
            next_h = from_a_to_h
            back[t, 0] = 1

        from_a_to_a = score_a
        from_h_to_a = score_h - switch_cost
        if from_a_to_a >= from_h_to_a:
            next_a = from_a_to_a
            back[t, 1] = 1
        else:
            next_a = from_h_to_a
            back[t, 1] = 0

        score_h = next_h + lp_h[t]
        score_a = next_a + lp_a[t]

    path[n - 1] = 0 if score_h >= score_a else 1
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _as_numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    return array.astype(np.float64, copy=False)


def _vector(value, name, n, probability=False):
    array = _as_numeric(value, name)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape (N,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if probability and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def _prediction_components(prediction, n):
    probability = None
    p_dead = None
    p_team = None

    if isinstance(prediction, Mapping):
        if "probability" in prediction:
            probability = _as_numeric(prediction["probability"], "probability")
        for key in ("p_dead", "pD", "pdead"):
            if key in prediction:
                p_dead = prediction[key]
                break
        for key in ("p_team", "pH", "pteam"):
            if key in prediction:
                p_team = prediction[key]
                break
    else:
        probability = _as_numeric(prediction, "probability")

    if probability is not None:
        if probability.shape != (n, 3):
            raise ValueError("probability must have shape (N, 3)")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("probability must be finite and nonnegative")
        mass = probability.sum(axis=1)
        if np.any(mass <= 0.0):
            raise ValueError("each probability row must have positive mass")
        joint = probability / mass[:, None]
        live_mass = joint[:, 0] + joint[:, 1]
        derived_team = np.full(n, 0.5, dtype=np.float64)
        np.divide(joint[:, 0], live_mass, out=derived_team, where=live_mass > 0.0)
        derived_dead = joint[:, 2]
    else:
        derived_team = None
        derived_dead = None

    if p_dead is None:
        if derived_dead is None:
            raise ValueError("prediction must provide probability or p_dead and p_team")
        p_dead = derived_dead
    else:
        p_dead = _vector(p_dead, "p_dead", n, probability=True)

    if p_team is None:
        if derived_team is None:
            raise ValueError("prediction must provide probability or p_dead and p_team")
        p_team = derived_team
    else:
        p_team = _vector(p_team, "p_team", n, probability=True)

    return p_dead, p_team


def _normalize_blocks(blocks, n):
    if blocks is None:
        if n == 0:
            return np.empty((0, 2), dtype=np.int64)
        raise ValueError("test must provide blocks")

    normalized = []
    for block in blocks:
        values = tuple(block)
        if len(values) == 2:
            start_value, end_value = values
        elif len(values) == 3:
            start_value, end_value = values[-2:]
        else:
            raise ValueError("each block must contain start and end indices")
        start = int(start_value)
        end = int(end_value)
        if start != start_value or end != end_value:
            raise ValueError("block indices must be integers")
        expected_start = normalized[-1][1] if normalized else 0
        if start != expected_start or end <= start or end > n:
            raise ValueError("blocks must partition observations in order")
        normalized.append((start, end))
    if (normalized[-1][1] if normalized else 0) != n:
        raise ValueError("blocks must cover every observation")
    return np.asarray(normalized, dtype=np.int64).reshape(-1, 2)


def _logit(value):
    clipped = np.clip(value, _EPS, 1.0 - _EPS)
    return np.log(clipped) - np.log1p(-clipped)


def dead_mask(p_dead, blocks, threshold=0.5, min_s=2.0, shift=0.0, fps=FPS):
    probabilities = _as_numeric(p_dead, "p_dead")
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("p_dead must be a finite vector")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("p_dead must be in [0, 1]")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("dead threshold must be in [0, 1]")
    if float(min_s) < 0.0 or float(fps) <= 0.0:
        raise ValueError("dead duration must be nonnegative and fps positive")

    ranges = _normalize_blocks(blocks, len(probabilities))
    shifted = 1.0 / (1.0 + np.exp(-(_logit(probabilities) + float(shift))))
    is_dead = shifted > float(threshold)
    min_frames = int(math.ceil(float(min_s) * float(fps)))
    mask = np.zeros(len(probabilities), dtype=bool)
    for start, end in ranges:
        cursor = start
        while cursor < end:
            if not is_dead[cursor]:
                cursor += 1
                continue
            run_start = cursor
            while cursor < end and is_dead[cursor]:
                cursor += 1
            if cursor - run_start >= min_frames:
                mask[run_start:cursor] = True
    return mask


@njit(cache=False)
def _count_possessions(codes, blocks):
    count = 0
    for start, end in blocks:
        previous = -1
        for index in range(start, end):
            state = int(codes[index])
            if state == 0 or state == 1:
                if state != previous:
                    count += 1
                previous = state
            else:
                previous = -1
    return count


def n_possessions(codes, blocks):
    labels = np.asarray(codes)
    if labels.ndim != 1:
        raise ValueError("codes must have shape (N,)")
    ranges = _normalize_blocks(blocks, len(labels))
    return _count_possessions(labels, ranges)


def calibrate_lambda(make_labels, blocks, target, lambdas=LAMBDA_GRID):
    """Choose nearest possession count on grid; count ties choose smaller lambda."""
    target = float(target)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target count must be finite and nonnegative")
    candidates = tuple(float(value) for value in lambdas)
    if not candidates or any(not math.isfinite(value) or value <= 0.0 for value in candidates):
        raise ValueError("lambda grid must contain positive finite values")

    best_lambda = None
    best_difference = math.inf
    ranges = np.asarray(blocks, dtype=np.int64).reshape(-1, 2)
    for candidate in candidates:
        count = _count_possessions(np.asarray(make_labels(candidate)), ranges)
        difference = abs(count - target)
        if difference < best_difference or (
            difference == best_difference and (best_lambda is None or candidate < best_lambda)
        ):
            best_lambda = candidate
            best_difference = difference
    return float(best_lambda)


def _team_evidence(p_team, X, test, config):
    clipped_team = np.clip(p_team, 1e-3, 1.0 - 1e-3)
    home = config["kappa"] * np.log(clipped_team)
    away = config["kappa"] * np.log1p(-clipped_team)

    if "distances" in test:
        distances = _as_numeric(test["distances"], "distances")
        if distances.shape != (len(X), 2):
            raise ValueError("distances must have shape (N, 2)")
        known = np.isfinite(distances).all(axis=1) & np.all(distances >= 0.0, axis=1)
    else:
        distances = X[:, :2]
        known = np.isfinite(distances).all(axis=1) & np.all(distances >= 0.0, axis=1)
        known &= np.all(distances < 30.0, axis=1)

    visible = np.isfinite(X[:, 3]) & (X[:, 3] >= 0.5)
    near = np.minimum(distances[:, 0], distances[:, 1])
    use = visible & known & (near <= config["gate_r"])
    if np.any(use):
        margin = np.clip(
            (distances[use, 1] - distances[use, 0]) / config["tau"],
            -10.0,
            10.0,
        )
        home[use] += -np.logaddexp(0.0, -margin)
        away[use] += -np.logaddexp(0.0, margin)
    return home, away


@njit(cache=False)
def _decode_team(home, away, blocks, lam, dead):
    codes = np.full(len(home), -1, dtype=np.int8)
    for start, end in blocks:
        if end > start:
            codes[start:end] = viterbi_2state(home[start:end], away[start:end], lam)
    codes[dead] = 2
    return codes


@njit(cache=False)
def _merge_short_runs(codes, blocks, min_frames):
    for block_start, block_end in blocks:
        cursor = block_start
        while cursor < block_end:
            if codes[cursor] == 2:
                cursor += 1
                continue

            live_end = cursor
            while live_end < block_end and codes[live_end] != 2:
                live_end += 1

            run_start = cursor
            previous_output = int(codes[run_start])
            first_run = True
            while run_start < live_end:
                label = int(codes[run_start])
                run_end = run_start + 1
                while run_end < live_end and int(codes[run_end]) == label:
                    run_end += 1
                if not first_run and run_end - run_start < min_frames:
                    codes[run_start:run_end] = previous_output
                else:
                    previous_output = label
                first_run = False
                run_start = run_end
            cursor = live_end
    return codes


def _resolve_config(config):
    if not isinstance(config, Mapping) or not isinstance(config.get("name"), str):
        raise ValueError("config must be a named configuration dictionary")
    resolved = dict(_DEFAULT_CONFIG)
    for key, value in config.items():
        if key == "name":
            continue
        if key not in resolved:
            raise ValueError(f"unknown decoder config field: {key}")
        resolved[key] = value

    for key in ("gate_r", "kappa", "merge_s", "min_dead_s", "rate_scale"):
        value = float(resolved[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
        resolved[key] = value
    threshold = float(resolved["dead_threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("dead_threshold must be in [0, 1]")
    resolved["dead_threshold"] = threshold
    for key in ("tau", "fixed_lambda"):
        value = float(resolved[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
        resolved[key] = value
    if not isinstance(resolved["shift"], (bool, np.bool_)):
        raise ValueError("shift must be boolean")
    if resolved["lambda_mode"] not in ("rate", "fixed"):
        raise ValueError("lambda_mode must be 'rate' or 'fixed'")
    return resolved


def decode(prediction, test, config, training_stats):
    """Decode H/A per block, overlay dead stretches, then merge short live runs."""
    if not isinstance(test, Mapping):
        raise ValueError("test must be an observation dictionary")
    if not isinstance(training_stats, Mapping):
        raise ValueError("training_stats must be a dictionary")
    resolved = _resolve_config(config)

    X = _as_numeric(test.get("X"), "test X")
    if X.ndim != 2 or X.shape[1] != 16:
        raise ValueError("test X must have shape (N, 16)")
    n = len(X)
    blocks = _normalize_blocks(test.get("blocks"), n)
    p_dead, p_team = _prediction_components(prediction, n)

    try:
        rate = float(training_stats["rate"])
        dead_share = float(training_stats["dead_share"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("training_stats must contain numeric rate and dead_share") from error
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("training rate must be finite and nonnegative")
    if not math.isfinite(dead_share) or not 0.0 <= dead_share <= 1.0:
        raise ValueError("training dead_share must be in [0, 1]")

    dead_shift = float(_logit(0.407) - _logit(dead_share)) if resolved["shift"] else 0.0
    dead = dead_mask(
        p_dead,
        blocks,
        threshold=resolved["dead_threshold"],
        min_s=resolved["min_dead_s"],
        shift=dead_shift,
    )
    home, away = _team_evidence(p_team, X, test, resolved)
    target_count = rate * sum(end - start for start, end in blocks) / (FPS * 60.0)
    target_count *= resolved["rate_scale"]

    if resolved["lambda_mode"] == "fixed":
        lam = resolved["fixed_lambda"]
    else:
        lam = calibrate_lambda(
            lambda value: _decode_team(home, away, blocks, value, dead),
            blocks,
            target_count,
        )

    codes = _decode_team(home, away, blocks, lam, dead)
    min_merge_frames = int(math.ceil(resolved["merge_s"] * FPS))
    _merge_short_runs(codes, blocks, min_merge_frames)
    actual_count = _count_possessions(codes, blocks)
    return {
        "codes": codes.astype(np.int8, copy=False),
        "lambda": float(lam),
        "target_count": float(target_count),
        "actual_count": int(actual_count),
        "dead_shift": float(dead_shift),
    }
