"""Permutation-invariant player-set tokens at 5 fps.

Each 5 fps step becomes (a) up to ``MAX_TOKENS`` player tokens with no identity and
(b) a small global vector. Inputs are observations only: positions, derived speed, the
team of each track slot and the ball when visible. Labels are attached separately and
never feed any feature. Pitch direction is normalised per half without labels: the side
whose players stand on the left on average is rotated to be the home side's half.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from exp1_common import FPS, Match, sample_indices

MAX_TOKENS = 28
TOKEN_NAMES = (
    "x", "y", "speed", "vx", "vy", "team", "dx_ball", "dy_ball", "proximity", "nearest",
)
GLOBAL_NAMES = (
    "ball_visible", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_outside",
    "since_ball", "until_ball", "n_home", "n_away", "home_speed", "away_speed",
)
# Mirror augmentation (swap teams + 180 degree rotation) acts on these columns.
TOKEN_NEGATE = (0, 1, 3, 4, 5, 6, 7)
GLOBAL_NEGATE = (1, 2, 3, 4)
GLOBAL_SWAP = ((8, 9), (10, 11))
VELOCITY_SPAN = 2  # frames either side at 25 fps
MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0
GAP_CAP_SECONDS = 120.0


@dataclass(frozen=True)
class TokenSet:
    tokens: np.ndarray  # (steps, MAX_TOKENS, len(TOKEN_NAMES)) float16
    mask: np.ndarray  # (steps, MAX_TOKENS) bool
    glob: np.ndarray  # (steps, len(GLOBAL_NAMES)) float16
    half: np.ndarray  # (steps,) int8 period of each step
    indices: np.ndarray  # (steps,) 25 fps frame index of each step


def _flip_signs(match: Match) -> np.ndarray:
    """+1/-1 per frame so that home players sit on the left half on average."""
    length = match.pitch[0]
    sign = np.ones(len(match.half), dtype="float32")
    home = match.teams == 0
    for period in np.unique(match.half):
        if period <= 0:
            continue
        index = np.flatnonzero(match.half == period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_x = np.nanmean(match.players[index][:, home, 0])
        sign[index] = -1.0 if np.isfinite(mean_x) and mean_x > length / 2 else 1.0
    return sign


def _velocity(values: np.ndarray, half: np.ndarray, limit: float) -> np.ndarray:
    """Central difference over +-VELOCITY_SPAN frames inside a period; invalid -> 0."""
    span = VELOCITY_SPAN
    output = np.zeros(values.shape, dtype="float32")
    for period in np.unique(half):
        if period <= 0:
            continue
        index = np.flatnonzero(half == period)
        v = values[index]
        if len(v) <= 2 * span:
            continue
        diff = (v[2 * span:] - v[: -2 * span]) * (FPS / (2 * span))
        speed = np.linalg.norm(diff, axis=-1, keepdims=True)
        diff = np.where(np.isfinite(speed) & (speed <= limit), diff, 0.0)
        output[index[span:-span]] = diff
    return output


def _gap_seconds(visible: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the ball was last seen and until it is seen again, per period."""
    since = np.full(len(visible), GAP_CAP_SECONDS, dtype="float32")
    until = np.full(len(visible), GAP_CAP_SECONDS, dtype="float32")
    for period in np.unique(half):
        if period <= 0:
            continue
        index = np.flatnonzero(half == period)
        n, seen = len(index), visible[index]
        position = np.arange(n)
        previous = np.maximum.accumulate(np.where(seen, position, -1))
        following = np.minimum.accumulate(np.where(seen, position, n)[::-1])[::-1]
        since[index] = np.where(previous >= 0, (position - previous) / FPS, GAP_CAP_SECONDS)
        until[index] = np.where(following < n, (following - position) / FPS, GAP_CAP_SECONDS)
    scale = np.log1p(GAP_CAP_SECONDS)
    return (
        np.log1p(np.minimum(since, GAP_CAP_SECONDS)) / scale,
        np.log1p(np.minimum(until, GAP_CAP_SECONDS)) / scale,
    )


def _select_slots(distance: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Up to MAX_TOKENS visible slots, nearest to the ball first (slot order if no ball)."""
    key = np.where(present, np.nan_to_num(distance, nan=0.0), np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    return order[:, :MAX_TOKENS]


def _ball_globals(match: Match, steps: np.ndarray, sign: np.ndarray) -> dict:
    length, width = match.pitch
    ball = match.ball
    visible = np.isfinite(ball).all(axis=1)
    velocity = _velocity(ball, match.half, MAX_BALL_SPEED)
    since, until = _gap_seconds(visible, match.half)
    b, s, vis = ball[steps], sign[steps], visible[steps]
    centred_x = np.where(vis, (b[:, 0] - length / 2) / (length / 2) * s, 0.0)
    centred_y = np.where(vis, (b[:, 1] - width / 2) / (width / 2) * s, 0.0)
    outside = vis & ((b[:, 0] < 0) | (b[:, 0] > length) | (b[:, 1] < 0) | (b[:, 1] > width))
    return {
        "ball_visible": vis.astype("float32"),
        "ball_x": centred_x,
        "ball_y": centred_y,
        "ball_vx": velocity[steps, 0] * s / 20.0,
        "ball_vy": velocity[steps, 1] * s / 20.0,
        "ball_outside": outside.astype("float32"),
        "since_ball": since[steps],
        "until_ball": until[steps],
    }


def build_tokens(match: Match) -> TokenSet:
    """Observation-only token set on the shared 5 fps grid."""
    steps = sample_indices(match.half)
    length, width = match.pitch
    sign = _flip_signs(match)
    players = match.players[steps]
    present = np.isfinite(players[:, :, 0]) & np.isfinite(players[:, :, 1])
    ball = match.ball[steps]
    ball_visible = np.isfinite(ball).all(axis=1)
    offset = players[:, :, :2] - ball[:, None, :]
    distance = np.where(ball_visible[:, None], np.linalg.norm(offset, axis=-1), np.nan)
    slots = _select_slots(distance, present)
    rows = np.arange(len(steps))[:, None]
    chosen_present = present[rows, slots]
    velocity = _velocity(match.players[:, :, :2], match.half, MAX_PLAYER_SPEED)[steps]
    s = sign[steps][:, None]
    team_sign = np.where(match.teams[slots] == 0, 1.0, -1.0)
    chosen_distance = distance[rows, slots]
    nearest = np.zeros(chosen_distance.shape, dtype="float32")
    has_ball = ball_visible & chosen_present.any(axis=1)
    masked = np.where(chosen_present, np.nan_to_num(chosen_distance, nan=np.inf), np.inf)
    nearest[np.flatnonzero(has_ball), masked[has_ball].argmin(axis=1)] = 1.0
    columns = [
        (players[rows, slots, 0] - length / 2) / (length / 2) * s,
        (players[rows, slots, 1] - width / 2) / (width / 2) * s,
        np.nan_to_num(players[rows, slots, 2]) / 8.0,
        velocity[rows, slots, 0] * s / 8.0,
        velocity[rows, slots, 1] * s / 8.0,
        team_sign,
        np.clip(np.nan_to_num(offset[rows, slots, 0]) * s / 20.0, -3, 3),
        np.clip(np.nan_to_num(offset[rows, slots, 1]) * s / 20.0, -3, 3),
        np.where(np.isfinite(chosen_distance), np.exp(-np.nan_to_num(chosen_distance) / 2), 0.0),
        nearest,
    ]
    tokens = np.stack(columns, axis=-1)
    tokens = np.where(chosen_present[..., None], np.nan_to_num(tokens), 0.0)
    glob = _ball_globals(match, steps, sign)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team, prefix in ((0, "home"), (1, "away")):
            side = match.teams == team
            glob["n_" + prefix] = present[:, side].sum(axis=1) / 11.0
            glob[prefix + "_speed"] = np.nan_to_num(np.nanmean(players[:, side, 2], axis=1)) / 8.0
    return TokenSet(
        tokens=tokens.astype("float16"),
        mask=chosen_present,
        glob=np.column_stack([glob[name] for name in GLOBAL_NAMES]).astype("float16"),
        half=match.half[steps].astype("int8"),
        indices=steps,
    )


def step_targets(labels: np.ndarray, token_set: TokenSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-step state (0 H, 1 A, 2 D, 3 ignore) and boundary target (dilated by one step)."""
    state = labels[token_set.indices].astype("int8")
    change = np.zeros(len(state), dtype=bool)
    same_half = token_set.half[1:] == token_set.half[:-1]
    valid = (state[1:] < 3) & (state[:-1] < 3) & same_half
    change[1:] = valid & (state[1:] != state[:-1])
    dilated = change.copy()
    dilated[:-1] |= change[1:]
    dilated[1:] |= change[:-1]
    return state, dilated.astype("int8")


def mirror(tokens: np.ndarray, glob: np.ndarray, state: np.ndarray) -> tuple:
    """Swap home/away and rotate 180 degrees; H<->A labels swap, D unchanged."""
    tokens, glob = tokens.copy(), glob.copy()
    tokens[..., TOKEN_NEGATE] *= -1
    glob[..., GLOBAL_NEGATE] *= -1
    for left, right in GLOBAL_SWAP:
        glob[..., [left, right]] = glob[..., [right, left]]
    swapped = np.where(state == 0, 1, np.where(state == 1, 0, state)).astype(state.dtype)
    return tokens, glob, swapped
