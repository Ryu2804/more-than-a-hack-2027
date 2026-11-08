"""Permutation-invariant features and train-only temporal model selection."""

import importlib.util
import sys
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from numba import njit
from lightgbm import LGBMClassifier

from experiment_data import Match, REPO, FPS

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "official_possession_metrics", REPO / "evaluation" / "possession_metrics.py"
)
PM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PM)
MODEL_FPS = 5
STRIDE = FPS // MODEL_FPS


def row_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    """Vectorized NaN percentile without a Python callback for every video frame."""
    sorted_values = np.sort(values, axis=1)
    count = np.isfinite(sorted_values).sum(axis=1)
    position = np.maximum(count - 1, 0) * percentile / 100
    low, high = np.floor(position).astype(int), np.ceil(position).astype(int)
    rows = np.arange(len(values))
    a, b = sorted_values[rows, low], sorted_values[rows, high]
    return np.where(count > 0, a + (b - a) * (position - low), np.nan)


def restart_features(match: Match, features: dict) -> dict:
    """Player configurations are restart evidence, never automatic D labels."""
    p, b = match.players, match.ball
    length, width = match.pitch
    x, y = b[:, 0], b[:, 1]
    close = np.linalg.norm(p[:, :, :2] - b[:, None], axis=2) < 3.0
    present = np.isfinite(p[:, :, 0])
    left = p[:, :, 0] < length / 2
    fractions = []
    for team in (0, 1):
        mask = present & (match.teams == team)
        fractions.append((left & mask).sum(1) / np.maximum(mask.sum(1), 1))
    split = np.maximum(
        fractions[0] * (1 - fractions[1]), fractions[1] * (1 - fractions[0])
    )
    in_goal = ((p[:, :, 0] < 5.5) | (p[:, :, 0] > length - 5.5)) & (
        np.abs(p[:, :, 1] - width / 2) < 9.16
    )
    on_line = ((p[:, :, 0] < 2.0) | (p[:, :, 0] > length - 2.0)) & (
        np.abs(p[:, :, 1] - width / 2) < 4.0
    )
    in_box = ((p[:, :, 0] < 16.5) | (p[:, :, 0] > length - 16.5)) & (
        np.abs(p[:, :, 1] - width / 2) < 20.16
    )
    spot = np.minimum(
        np.hypot(x - 11.0, y - width / 2), np.hypot(x - length + 11.0, y - width / 2)
    )
    near_line = (p[:, :, 1] < 1.5) | (p[:, :, 1] > width - 1.5)
    cues = {
        "restart_kickoff": (features["centre_distance"] < 2.0)
        & (split > 0.7)
        & close.any(1),
        "restart_penalty": (spot < 2.0)
        & on_line.any(1)
        & (in_box.sum(1) <= 3)
        & close.any(1),
        "restart_goal_kick": ((x < 5.5) | (x > length - 5.5))
        & (np.abs(y - width / 2) < 9.16)
        & (close & in_goal).any(1),
        "restart_corner": (features["corner_distance"] < 2.0) & close.any(1),
        "restart_throw_in": (features["touchline_distance"] < 1.5)
        & (close & near_line).any(1),
    }
    any_specific = np.column_stack(list(cues.values())).any(1)
    cues["restart_free_kick"] = (
        (features["restart_static"] > 0) & ~any_specific & close.any(1)
    )
    any_cue = np.column_stack(list(cues.values())).any(1)
    cues["restart_team"] = np.where(any_cue, features["nearest_team"], 0.0)
    return {name: np.asarray(value, dtype="float32") for name, value in cues.items()}


def outer_splits(names: list[str]):
    for held in names:
        yield [name for name in names if name != held], held


def score(truth: np.ndarray, prediction: np.ndarray) -> dict:
    runs = PM.valid_runs(truth)
    filled = PM.fill_abstentions(prediction, runs)
    spells = PM.spell_scores(truth, filled, runs)
    return {
        "spell_f1": spells["mean_f1"],
        "macro_f1": PM.macro_f1(PM.scored_confusion(PM.confusion(truth, prediction))),
        "n_true": spells["n_gt"],
        "n_pred": spells["n_pred"],
        "count_error_ratio": (
            (spells["n_pred"] / spells["n_gt"] - 1) if spells["n_gt"] else 0.0
        ),
    }


def _filter(values: np.ndarray, half: np.ndarray, seconds: float) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype="float32")
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        v = values[index]
        finite = np.isfinite(v)
        size = max(1, round(seconds * FPS))
        numerator = uniform_filter1d(
            np.where(finite, v, 0.0).astype("float64"), size, axis=0, mode="nearest"
        )
        denominator = uniform_filter1d(
            finite.astype("float64"), size, axis=0, mode="nearest"
        )
        output[index] = np.divide(
            numerator,
            denominator,
            out=np.full(v.shape, np.nan),
            where=denominator > 1e-8,
        )
    return output


def _gap_context(
    values: np.ndarray, visible: np.ndarray, half: np.ndarray
) -> tuple[np.ndarray, ...]:
    before, after = np.full_like(values, np.nan), np.full_like(values, np.nan)
    since, until = np.full(len(values), 120.0), np.full(len(values), 120.0)
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        n = len(index)
        prev = np.maximum.accumulate(np.where(visible[index], np.arange(n), -1))
        nex = np.minimum.accumulate(np.where(visible[index], np.arange(n), n)[::-1])[
            ::-1
        ]
        keep = prev >= 0
        before[index[keep]] = values[index[prev[keep]]]
        since[index[keep]] = (np.arange(n)[keep] - prev[keep]) / FPS
        keep = nex < n
        after[index[keep]] = values[index[nex[keep]]]
        until[index[keep]] = (nex[keep] - np.arange(n)[keep]) / FPS
    return before, after, np.minimum(since, 120), np.minimum(until, 120)


def feature_matrix(
    match: Match, context: bool = True, deadball: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Observations only; labels, player identity and source annotations never read."""
    p, ball, teams, half = match.players, match.ball, match.teams, match.half
    features = {}
    visible = np.isfinite(ball).all(axis=1)
    features["visible"] = visible.astype("float32")
    features["ball_x"], features["ball_y"] = (
        ball[:, 0] / match.pitch[0],
        ball[:, 1] / match.pitch[1],
    )
    distance = np.linalg.norm(p[:, :, :2] - ball[:, None, :], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team, prefix in ((0, "home_"), (1, "away_")):
            slots = teams == team
            xy, speed, d = p[:, slots, :2], p[:, slots, 2], distance[:, slots]
            features[prefix + "count"] = np.isfinite(xy[:, :, 0]).sum(axis=1)
            for axis, name, scale in (
                (0, "x", match.pitch[0]),
                (1, "y", match.pitch[1]),
            ):
                features[prefix + "mean_" + name] = (
                    np.nanmean(xy[:, :, axis], axis=1) / scale
                )
                features[prefix + "std_" + name] = (
                    np.nanstd(xy[:, :, axis], axis=1) / scale
                )
                for percentile in (10, 90):
                    features[prefix + str(percentile) + "_" + name] = (
                        row_percentile(xy[:, :, axis], percentile) / scale
                    )
            features[prefix + "speed"] = np.nanmean(speed, axis=1)
            features[prefix + "slow"] = np.nanmean(
                np.where(np.isfinite(speed), speed < 1.5, np.nan), axis=1
            )
            nearest = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
            for rank in range(3):
                features[prefix + "distance" + str(rank)] = np.where(
                    np.isfinite(nearest[:, rank]), nearest[:, rank], np.nan
                )
            features[prefix + "near3"] = np.sum(d < 3.0, axis=1)
            features[prefix + "near8"] = np.sum(d < 8.0, axis=1)
    dh, da = features["home_distance0"], features["away_distance0"]
    features["margin"] = da - dh
    features["nearest_team"] = np.where(
        np.isfinite(dh + da), np.where(dh <= da, 1.0, -1.0), np.nan
    )
    features["collective_speed"] = (features["home_speed"] + features["away_speed"]) / 2
    # Do not differentiate across half boundaries, missing samples or long gaps.
    delta = np.full_like(ball, np.nan)
    valid = visible[1:] & visible[:-1] & (half[1:] == half[:-1])
    delta[1:] = np.where(valid[:, None], np.diff(ball, axis=0) * FPS, np.nan)
    features["ball_motion"] = np.minimum(np.linalg.norm(delta, axis=1), 50)
    if context:
        for name in (
            "margin",
            "nearest_team",
            "collective_speed",
            "home_speed",
            "away_speed",
            "visible",
            "ball_motion",
        ):
            for seconds in (1.0, 3.0, 8.0):
                features[name + "_mean" + str(int(seconds))] = _filter(
                    features[name], half, seconds
                )
        edge_values = np.column_stack(
            [features["nearest_team"], features["margin"], ball]
        )
        before, after, since, until = _gap_context(edge_values, visible, half)
        for j, name in enumerate(("team", "margin", "x", "y")):
            features["previous_" + name], features["next_" + name] = (
                before[:, j],
                after[:, j],
            )
        features["since_ball"], features["until_ball"] = since, until
        features["gap_duration"] = since + until
    if deadball:
        x, y = ball[:, 0], ball[:, 1]
        length, width = match.pitch
        features["outside"] = np.where(
            visible,
            ((x < 0) | (x > length) | (y < 0) | (y > width)).astype("float32"),
            np.nan,
        )
        features["touchline_distance"] = np.minimum(np.abs(y), np.abs(width - y))
        features["goalline_distance"] = np.minimum(np.abs(x), np.abs(length - x))
        features["centre_distance"] = np.hypot(x - length / 2, y - width / 2)
        features["corner_distance"] = np.hypot(
            features["touchline_distance"], features["goalline_distance"]
        )
        for name in ("outside", "corner_distance", "centre_distance", "ball_motion"):
            features[name + "_restart_context"] = _filter(features[name], half, 3.0)
        features["restart_static"] = (features["ball_motion"] < 0.5).astype(
            "float32"
        ) * (features["collective_speed"] < 1.5)
        features.update(restart_features(match, features))
    names = list(features)
    return np.column_stack([features[name] for name in names]).astype("float32"), names


@njit(cache=True)
def _viterbi(
    logp: np.ndarray,
    penalty_team: float,
    penalty_dead: float,
    minimum: int,
    restart_hint: np.ndarray,
    restart_weight: float,
) -> np.ndarray:
    """Expanded states enforce <=1s minimum duration, with free start/end edges."""
    n, states = len(logp), 3 * minimum
    costs = np.full(states, -1e30)
    costs[minimum - 1], costs[2 * minimum - 1], costs[3 * minimum - 1] = (
        logp[0, 0],
        logp[0, 1],
        logp[0, 2],
    )
    back = np.zeros((n, states), dtype=np.int16)
    for t in range(1, n):
        new = np.full(states, -1e30)
        for state in range(states):
            label, age = state // minimum, state % minimum
            if age > 0:
                old = state - 1
                new[state], back[t, state] = costs[old] + logp[t, label], old
            if age == minimum - 1:
                old = state
                value = costs[old] + logp[t, label]
                if value > new[state]:
                    new[state], back[t, state] = value, old
            if age == 0:
                for old_label in range(3):
                    if old_label == label:
                        continue
                    old = (old_label + 1) * minimum - 1
                    penalty = (
                        penalty_dead if label == 2 or old_label == 2 else penalty_team
                    )
                    if old_label == 2 and label < 2:
                        side = 1.0 if label == 0 else -1.0
                        if restart_hint[t] * side < 0:
                            penalty += restart_weight
                    value = costs[old] - penalty + logp[t, label]
                    if value > new[state]:
                        new[state], back[t, state] = value, old
        costs = new
    state = np.argmax(costs)
    result = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        result[t] = state // minimum
        state = back[t, state]
    return result


def decode(
    probabilities: np.ndarray,
    half: np.ndarray,
    team_penalty: float = 3.0,
    dead_penalty: float = 3.0,
    minimum: int = 1,
    fps: int = MODEL_FPS,
    restart_hint: np.ndarray | None = None,
    restart_weight: float = 0.0,
) -> np.ndarray:
    if len(probabilities) == 0 or probabilities.shape != (len(half), 3):
        raise ValueError(
            "Expected nonempty three-class probabilities and matching periods"
        )
    if minimum < 1 or minimum > fps:
        raise ValueError("Minimum duration must be between one frame and one second")
    output = np.full(len(half), 3, dtype="int8")
    hint = np.zeros(len(half)) if restart_hint is None else np.nan_to_num(restart_hint)
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        output[index] = _viterbi(
            np.log(np.clip(probabilities[index], 1e-6, 1.0)),
            team_penalty,
            dead_penalty,
            minimum,
            hint[index],
            restart_weight,
        )
    return output


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=160,
        num_leaves=23,
        learning_rate=0.07,
        min_child_samples=120,
        max_bin=127,
        reg_lambda=2.0,
        n_jobs=4,
        random_state=20261108,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def calibrate(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probabilities, 1e-6, 1.0)) / temperature
    values = np.exp(logp - logp.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def paired_bootstrap(gains: list[float], draws: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(20261108)
    samples = rng.choice(np.asarray(gains), (draws, len(gains)), replace=True).mean(
        axis=1
    )
    return tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))
