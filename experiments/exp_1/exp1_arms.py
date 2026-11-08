"""Reference arms re-run on the identical frozen proxy for a like-for-like comparison.

* baseline      - organisers' starter rule: nearest player + +-50 frame majority (frozen judge).
* rasya_c06     - rasya 012 deployable: touch-hold (r 0.8 m, minlen 30) + dead cue
                  out_or_slow2 (ball out >= 15 frames or nearest player < 2 m/s >= 25 frames)
                  + merge dead runs < 150 frames. Parameters frozen as rasya selected them.
* rasya_c08     - same with the long-slow dead cue (slow run >= 400 frames).
* gbm           - exp_0 family 'deadball': hand-crafted features + LightGBM (frozen snapshot code).
None of these are tuned on the proxy here; the GBM decoder is selected inside training folds.
"""

from __future__ import annotations

import numpy as np

from exp1_common import JD, JM, MATCH_IDS, ROOT, Match, sample_indices

TOUCH_RADIUS = 0.8
TOUCH_MIN_FRAMES = 30
OUT_MARGIN = 0.1
OUT_RUN = 15
SLOW_SPEED = 2.0
SLOW_RUN_C06 = 25
SLOW_RUN_C08 = 400
DEAD_MERGE = 150
GBM_DIR = ROOT / "cache" / "gbm_features"


def nearest_player(match: Match) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: distance, team (-1 none) and speed of the player nearest the ball."""
    distance = np.linalg.norm(match.players[:, :, :2] - match.ball[:, None, :], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    index = distance.argmin(axis=1)
    rows = np.arange(len(index))
    dmin = distance[rows, index]
    answered = np.isfinite(dmin)
    team = np.where(answered, match.teams[index], -1).astype("int8")
    speed = np.where(answered, match.players[rows, index, 2], np.nan)
    return dmin, team, speed


def baseline_arm(match: Match) -> np.ndarray:
    _, team, _ = nearest_player(match)
    return JD.smooth_baseline(team, match.half, 50)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, flags, False].astype("int8"))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def run_mask(flags: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros(len(flags), dtype=bool)
    for start, stop in _runs(flags):
        if stop - start >= min_len:
            out[start:stop] = True
    return out


def touch_hold(dmin: np.ndarray, team: np.ndarray) -> np.ndarray:
    """rasya touch.hold: carry the team of the last touch; merge segments < minlen."""
    touch = np.flatnonzero((dmin < TOUCH_RADIUS) & (team >= 0))
    out = np.full(len(team), -1, dtype="int8")
    if len(touch):
        position = np.searchsorted(touch, np.arange(len(team)), side="right") - 1
        out = np.where(position >= 0, team[touch][np.clip(position, 0, None)], -1).astype("int8")
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    ends = np.r_[starts[1:], len(out)]
    for a, b in zip(starts, ends):
        if b - a < TOUCH_MIN_FRAMES:
            if a > 0:
                out[a:b] = out[a - 1]
            elif b < len(out):
                out[a:b] = out[b]
    return out


def merge_short_dead(series: np.ndarray, k: int) -> np.ndarray:
    out = series.copy()
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    ends = np.r_[starts[1:], len(out)]
    for a, b in zip(starts, ends):
        if out[a] == 2 and (b - a) < k:
            out[a:b] = out[a - 1] if a > 0 else (out[b] if b < len(out) else 2)
    return out


def rasya_arm(match: Match, slow_run: int) -> np.ndarray:
    dmin, team, speed = nearest_player(match)
    answered = np.isfinite(dmin)
    length, width = match.pitch
    x, y = match.ball[:, 0], match.ball[:, 1]
    outside = answered & (
        (x < -OUT_MARGIN) | (x > length + OUT_MARGIN) | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN)
    )
    slow = answered & np.isfinite(speed) & (speed < SLOW_SPEED)
    dead = run_mask(outside, OUT_RUN) | run_mask(slow, slow_run)
    held = touch_hold(dmin, team)
    base = np.where(dead, 2, np.where(held >= 0, held, 2)).astype("int8")
    prediction = merge_short_dead(base, DEAD_MERGE)
    return np.where(match.half > 0, prediction, 3).astype("int8")


def gbm_features(name: str, match: Match | None = None) -> dict:
    """exp_0 'deadball' feature family on the 5 fps grid, cached locally (proxy only)."""
    path = GBM_DIR / f"{name}.npz"
    if match is None and path.exists():
        with np.load(path) as z:
            return {key: z[key] for key in z.files}
    if match is None:
        raise ValueError("Match required to build features")
    x, names = JM.feature_matrix(match, True, True)
    steps = sample_indices(match.half)
    result = {"x": x[steps], "names": np.array(names)}
    if name in MATCH_IDS:
        GBM_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **result)
    return result


def gbm_fit(train_x: list[np.ndarray], train_y: list[np.ndarray]):
    """exp_0 LightGBM settings; every second 5 fps step as in exp_0 (2.5 fps fit)."""
    x = np.concatenate([v[::2] for v in train_x])
    y = np.concatenate([v[::2] for v in train_y])
    keep = y < 3
    model = JM.make_model()
    model.fit(x[keep], y[keep])
    return model


def gbm_predict(model, x: np.ndarray) -> np.ndarray:
    return model.booster_.predict(x, num_threads=4).astype("float32")
