from collections.abc import Mapping
import math

import numpy as np

from numba import njit

from support import base

_previous = base.import_path('wave5_base_decoder', base.HERE / 'decoder.py')
FPS = _previous.FPS
LAMBDA_GRID = _previous.LAMBDA_GRID
_DEFAULT_CONFIG = {**_previous._DEFAULT_CONFIG, 'shift': False}
viterbi_2state = _previous.viterbi_2state
_as_numeric = _previous._as_numeric
_vector = _previous._vector
_prediction_components = _previous._prediction_components
_normalize_blocks = _previous._normalize_blocks
_logit = _previous._logit
dead_mask = _previous.dead_mask
_count_possessions = _previous._count_possessions
n_possessions = _previous.n_possessions
calibrate_lambda = _previous.calibrate_lambda
_team_evidence = _previous._team_evidence
_decode_team = _previous._decode_team
_merge_short_runs = _previous._merge_short_runs

CONFIGS = [
    {"name": "wave4", **_DEFAULT_CONFIG},
    {"name": "postmerge", **_DEFAULT_CONFIG},
    {"name": "live_reset", **_DEFAULT_CONFIG},
    {"name": "boundary_reset", **_DEFAULT_CONFIG},
    {"name": "dead1", **{**_DEFAULT_CONFIG, "min_dead_s": 1.0}},
    {"name": "dead45", **{**_DEFAULT_CONFIG, "dead_threshold": 0.45, "min_dead_s": 1.0}},
    {"name": "dead55", **{**_DEFAULT_CONFIG, "dead_threshold": 0.55}},
    {"name": "no_merge", **{**_DEFAULT_CONFIG, "merge_s": 0.0}},
]




@njit(cache=False)
def viterbi_2state_variable(lp_h, lp_a, switch_costs):
    """Use switch_costs[t] for transition into t; index zero is unused."""
    n = len(lp_h)
    path = np.empty(n, dtype=np.int8)
    if n == 0:
        return path

    back = np.empty((n, 2), dtype=np.int8)
    score_h = lp_h[0]
    score_a = lp_a[0]
    for t in range(1, n):
        switch_cost = switch_costs[t]
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


@njit(cache=False)
def _decode_live_team(home, away, blocks, dead, transition_costs, lam):
    codes = np.full(len(home), -1, dtype=np.int8)
    for block_start, block_end in blocks:
        cursor = block_start
        while cursor < block_end:
            if dead[cursor]:
                cursor += 1
                continue
            live_start = cursor
            while cursor < block_end and not dead[cursor]:
                cursor += 1
            if len(transition_costs) == 0:
                codes[live_start:cursor] = viterbi_2state(
                    home[live_start:cursor], away[live_start:cursor], lam
                )
            else:
                codes[live_start:cursor] = viterbi_2state_variable(
                    home[live_start:cursor],
                    away[live_start:cursor],
                    transition_costs[live_start:cursor],
                )
    codes[dead] = 2
    return codes




def _resolve_config(config):
    if not isinstance(config, Mapping) or not isinstance(config.get("name"), str):
        raise ValueError("config must be a named configuration dictionary")
    name = config["name"]
    if name not in {value["name"] for value in CONFIGS}:
        raise ValueError(f"unknown decoder config: {name}")

    resolved = _previous._resolve_config(config)
    if resolved["shift"]:
        raise ValueError("wave5 decoder does not shift dead probabilities")
    if resolved["lambda_mode"] != "rate":
        raise ValueError("wave5 decoder calibrates over the full rate lambda grid")
    return resolved


def _boundary_scores(prediction, p_team, blocks, dead):
    has_boundary = False
    if isinstance(prediction, Mapping) and "has_boundary" in prediction:
        value = prediction["has_boundary"]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("has_boundary must be boolean")
        has_boundary = bool(value)

    if has_boundary:
        if "boundary" not in prediction:
            raise ValueError("prediction with has_boundary=True must provide boundary")
        scores = _vector(prediction["boundary"], "boundary", len(p_team), probability=True).copy()
    else:
        scores = np.zeros(len(p_team), dtype=np.float64)
        for block_start, block_end in blocks:
            cursor = block_start
            while cursor < block_end:
                if dead[cursor]:
                    cursor += 1
                    continue
                live_start = cursor
                while cursor < block_end and not dead[cursor]:
                    cursor += 1
                if cursor - live_start > 1:
                    changes = np.abs(np.diff(_logit(p_team[live_start:cursor]))) / 4.0
                    scores[live_start + 1:cursor] = np.clip(changes, 0.0, 1.0)

    for block_start, block_end in blocks:
        cursor = block_start
        while cursor < block_end:
            if dead[cursor]:
                cursor += 1
                continue
            scores[cursor] = 0.0
            while cursor < block_end and not dead[cursor]:
                cursor += 1
    return scores


def decode(prediction, test, config, training_stats):
    """Decode possession labels with wave4, post-merge, and live-reset controls."""
    if not isinstance(test, Mapping):
        raise ValueError("test must be an observation dictionary")
    if not isinstance(training_stats, Mapping):
        raise ValueError("training_stats must be a dictionary")
    name = config.get("name") if isinstance(config, Mapping) else None
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

    dead_shift = 0.0
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

    live_reset = name in {"live_reset", "boundary_reset", "dead1", "dead45", "dead55", "no_merge"}
    use_boundary = name == "boundary_reset"
    if use_boundary:
        boundary_scores = _boundary_scores(prediction, p_team, blocks, dead)
        transition_weights = 1.0 - 0.5 * boundary_scores
        transition_costs = np.empty(n, dtype=np.float64)
    else:
        transition_weights = None
        transition_costs = np.empty(0, dtype=np.float64)
    min_merge_frames = int(math.ceil(resolved["merge_s"] * FPS))
    calibrate_after_merge = name != "wave4"

    def make_labels(candidate):
        if live_reset:
            if use_boundary:
                np.multiply(transition_weights, candidate, out=transition_costs)
            labels = _decode_live_team(home, away, blocks, dead, transition_costs, candidate)
        else:
            labels = _decode_team(home, away, blocks, candidate, dead)
        if calibrate_after_merge:
            labels = _merge_short_runs(labels, blocks, min_merge_frames)
        return labels

    lam = calibrate_lambda(make_labels, blocks, target_count)
    codes = make_labels(lam)
    if not calibrate_after_merge:
        codes = _merge_short_runs(codes, blocks, min_merge_frames)
    actual_count = _count_possessions(codes, blocks)
    return {
        "codes": codes.astype(np.int8, copy=False),
        "lambda": float(lam),
        "target_count": float(target_count),
        "actual_count": int(actual_count),
        "dead_shift": float(dead_shift),
    }
