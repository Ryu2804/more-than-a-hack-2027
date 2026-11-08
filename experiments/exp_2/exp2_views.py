"""Observation transforms for exp_2 (fit-relevant: hashed into the prediction cache key).

* ``target_like`` - label-free stress view of a held-out match (PREREG, "Views"):
  off-pitch ball kept visible only for the first ``K_OUT`` frames of each off-pitch run,
  and hidden on long slow-play runs found from collective player speed.
* ``target_aug``  - training augmentation (uses the *training* match's labels):
  off-pitch tails hidden after a random ``k`` per run, plus ball hidden inside a random
  half of the long dead spells.

Both only ever remove ball observations; players, teams, halves and labels are unchanged.
Both commute with the home/away + 180 degree mirror used for training augmentation.
"""

from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

from exp1_common import BASE_SEED, JD, MATCH_IDS, Match

K_OUT = 37  # frames; label-free calibration: pooled off-pitch share of visible = 4.2 % (BAR-ATH)
SLOW_SPEED = 1.0  # m/s, smoothed collective speed (team EDA: 0.6-1.5 m/s before restarts)
SLOW_WINDOW = 51  # frames (2.04 s) moving average inside each period
SLOW_MIN_FRAMES = 125  # 5 s
NAN_SPEED = 2.0  # frozen degrader's fill value for frames without players
AUG_ROUNDS = (1, 2)  # corruption rounds that receive target_aug (round 0 stays original)
AUG_K_MAX = 74  # k ~ U{0..74}, mean 37 = K_OUT
AUG_DEAD_MIN = 125  # dead spells of at least 5 s
AUG_DEAD_PROB = 0.5
AUG_DEAD_MARGIN = 25  # keep 1 s visible at both ends of a hidden dead spell
AUG_SEED_OFFSET = 7_000_000


def graded_runs(mask: np.ndarray, half: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, stop) runs of ``mask`` inside graded periods (never across halves)."""
    return [(int(index[a]), int(index[b - 1]) + 1)
            for period in np.unique(half) if period > 0
            for index in [np.flatnonzero(half == period)]
            for a, b in JD.runs(mask[index])]


def ball_visible(match: Match) -> np.ndarray:
    return np.isfinite(match.ball).all(axis=1)


def outside_mask(match: Match) -> np.ndarray:
    """Visible ball strictly outside the pitch rectangle (no margin), graded frames only."""
    length, width = match.pitch
    x, y = match.ball[:, 0], match.ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = (x < 0) | (x > length) | (y < 0) | (y > width)
    return ball_visible(match) & outside & (match.half > 0)


def collective_speed(match: Match) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(match.players[:, :, 2], axis=1)
    return np.nan_to_num(speed, nan=NAN_SPEED)


def slow_mask(match: Match) -> np.ndarray:
    """Frames inside runs of smoothed collective speed < SLOW_SPEED lasting >= SLOW_MIN_FRAMES."""
    speed = collective_speed(match)
    smooth = np.full(len(speed), np.inf)
    for period in np.unique(match.half):
        if period <= 0:
            continue
        index = np.flatnonzero(match.half == period)
        smooth[index] = uniform_filter1d(speed[index], min(len(index), SLOW_WINDOW))
    hide = np.zeros(len(speed), dtype=bool)
    for start, stop in graded_runs(smooth < SLOW_SPEED, match.half):
        if stop - start >= SLOW_MIN_FRAMES:
            hide[start:stop] = True
    return hide


def _hide_ball(match: Match, hide: np.ndarray) -> Match:
    ball = match.ball.copy()
    ball[hide] = np.nan
    return replace(match, ball=ball)


def outside_tail_mask(match: Match, keep: np.ndarray | int) -> np.ndarray:
    """Hide each off-pitch run after its first ``keep`` frames (scalar or one value per run)."""
    runs = graded_runs(outside_mask(match), match.half)
    keeps = np.broadcast_to(np.asarray(keep), (len(runs),))
    hide = np.zeros(len(match.half), dtype=bool)
    for (start, stop), k in zip(runs, keeps):
        hide[start + int(k):stop] = True
    return hide


def target_like(match: Match) -> Match:
    """Label-free, deterministic stress view (never reads ``match.labels``)."""
    hide = outside_tail_mask(match, K_OUT) | slow_mask(match)
    return _hide_ball(match, hide)


def aug_seed(name: str, round_index: int) -> int:
    return BASE_SEED + 1000 * round_index + MATCH_IDS.index(name) + AUG_SEED_OFFSET


def target_aug(match: Match, rng: np.random.Generator) -> Match:
    """Training augmentation; uses the training match's labels to place dead-spell hiding."""
    if match.labels is None:
        raise ValueError("target_aug needs a labelled training match")
    runs = graded_runs(outside_mask(match), match.half)
    hide = outside_tail_mask(match, rng.integers(0, AUG_K_MAX + 1, size=len(runs)))
    dead_runs = graded_runs(match.labels == 2, match.half)
    chosen = rng.random(len(dead_runs)) < AUG_DEAD_PROB
    for (start, stop), pick in zip(dead_runs, chosen):
        if pick and stop - start >= AUG_DEAD_MIN:
            hide[start + AUG_DEAD_MARGIN:stop - AUG_DEAD_MARGIN] = True
    return _hide_ball(match, hide)


def mirror_match(match: Match) -> Match:
    """Swap home/away and rotate 180 degrees at the Match level (labels H<->A)."""
    length, width = match.pitch
    flip = np.array([length, width], dtype=match.ball.dtype)
    players = match.players.copy()
    players[:, :, :2] = flip - players[:, :, :2]
    labels = None if match.labels is None else np.where(
        match.labels == 0, 1, np.where(match.labels == 1, 0, match.labels)).astype(match.labels.dtype)
    return replace(match, players=players, ball=flip - match.ball,
                   teams=(1 - match.teams).astype(match.teams.dtype), labels=labels)
