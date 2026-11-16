"""Permutation-invariant player tokens and global features on the 5 fps grid."""

from __future__ import annotations


import numpy as np

MAX_TOKENS = 28
TOKEN_NAMES = (
    "x", "y", "speed", "vx", "vy", "team", "dx_ball", "dy_ball", "proximity", "nearest",
)
GLOBAL_NAMES = (
    "ball_visible", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_outside",
    "since_ball", "until_ball", "n_home", "n_away", "home_speed", "away_speed",
    "control_d_home", "control_d_away", "control_margin", "control_ball_visible",
    "control_ball_boundary_distance", "control_gap_age", "control_gap_remaining",
)
TOKEN_NEGATE = (0, 1, 3, 4, 5, 6, 7)
GLOBAL_NEGATE = (1, 2, 3, 4, 14)
GLOBAL_SWAP = ((8, 9), (10, 11), (12, 13))
FPS = 5.0
PITCH_SCALE = np.asarray((105.3, 68.0), dtype=np.float32)
MAX_SPEED = 12.0
GAP_CAP_SECONDS = 120.0



def _block_pairs(blocks):
    return [(int(begin), int(end)) for begin, end in np.asarray(blocks, dtype=np.int64).reshape(-1, 2)]


def _orientation(players, teams, blocks):
    sign = np.ones(len(players), dtype=np.float32)
    home = teams == 0
    for begin, end in blocks:
        home_x = players[begin:end, home, 0]
        finite_home_x = home_x[np.isfinite(home_x)]
        if finite_home_x.size and float(np.mean(finite_home_x)) > 0.5:
            sign[begin:end] = -1.0
    return sign


def _velocity(values, blocks, limit=MAX_SPEED):
    output = np.zeros(values.shape, dtype=np.float32)
    for begin, end in blocks:
        if end - begin < 3:
            continue
        delta = (values[begin + 2:end] - values[begin:end - 2]) * PITCH_SCALE
        velocity = delta * (FPS / 2.0)
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        valid = np.isfinite(speed) & (speed <= limit)
        output[begin + 1:end - 1] = np.where(valid, velocity, 0.0)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def _ball_gaps(visible, blocks):
    since = np.full(len(visible), GAP_CAP_SECONDS, dtype=np.float32)
    until = np.full(len(visible), GAP_CAP_SECONDS, dtype=np.float32)
    for begin, end in blocks:
        count = end - begin
        position = np.arange(count)
        seen = visible[begin:end]
        previous = np.maximum.accumulate(np.where(seen, position, -1))
        following = np.minimum.accumulate(np.where(seen, position, count)[::-1])[::-1]
        since[begin:end] = np.where(
            previous >= 0, (position - previous) / FPS, GAP_CAP_SECONDS
        )
        until[begin:end] = np.where(
            following < count, (following - position) / FPS, GAP_CAP_SECONDS
        )
    scale = np.log1p(GAP_CAP_SECONDS)
    return (
        np.log1p(np.minimum(since, GAP_CAP_SECONDS)) / scale,
        np.log1p(np.minimum(until, GAP_CAP_SECONDS)) / scale,
    )




def _select_slots(present, x, y, team, speed, vx, vy, distance, ball_visible):
    slots = np.flatnonzero(present)
    if not len(slots):
        return slots
    if ball_visible:
        order = np.lexsort((speed[slots], vy[slots], vx[slots], team[slots], y[slots], x[slots], distance[slots]))
    else:
        order = np.lexsort((speed[slots], vy[slots], vx[slots], team[slots], y[slots], x[slots]))
    return slots[order[:MAX_TOKENS]]


def build(data):
    players = np.asarray(data["players"], dtype=np.float32)
    teams = np.asarray(data["teams"])
    ball = np.asarray(data["ball"], dtype=np.float32)
    blocks = _block_pairs(data["blocks"])
    n, slot_count = players.shape[:2]

    known_team = (teams == 0) | (teams == 1)
    players = np.where(np.isfinite(players), players, np.nan).astype(np.float32, copy=False)
    if slot_count:
        players[:, ~known_team] = np.nan
    ball = np.where(np.isfinite(ball), ball, np.nan).astype(np.float32, copy=False)
    present = np.isfinite(players).all(axis=2) & known_team[None, :]
    ball_visible = np.isfinite(ball).all(axis=1)
    sign = _orientation(players, teams, blocks)

    players_m = players * PITCH_SCALE
    ball_m = ball * PITCH_SCALE
    player_velocity = _velocity(players, blocks)
    ball_velocity = _velocity(ball, blocks, limit=40.0)
    velocity_speed = np.linalg.norm(player_velocity, axis=2)
    centered = (players - 0.5) * 2.0 * sign[:, None, None]
    oriented_velocity = player_velocity * sign[:, None, None]
    team_sign = np.where(teams == 0, 1.0, -1.0).astype(np.float32)

    tokens = np.zeros((n, MAX_TOKENS, len(TOKEN_NAMES)), dtype=np.float32)
    mask = np.zeros((n, MAX_TOKENS), dtype=bool)
    for frame in range(n):
        frame_present = present[frame]
        if not frame_present.any():
            continue
        frame_distance = np.full(slot_count, np.inf, dtype=np.float32)
        if ball_visible[frame]:
            delta = players_m[frame] - ball_m[frame]
            frame_distance = np.linalg.norm(delta, axis=1)
            frame_distance[~frame_present] = np.inf
        selected = _select_slots(
            frame_present,
            centered[frame, :, 0],
            centered[frame, :, 1],
            team_sign,
            velocity_speed[frame],
            oriented_velocity[frame, :, 0],
            oriented_velocity[frame, :, 1],
            frame_distance,
            bool(ball_visible[frame]),
        )
        selected_count = len(selected)
        mask[frame, :selected_count] = True
        tokens[frame, :selected_count, 0:2] = centered[frame, selected]
        tokens[frame, :selected_count, 2] = velocity_speed[frame, selected] / 8.0
        tokens[frame, :selected_count, 3:5] = oriented_velocity[frame, selected] / 8.0
        tokens[frame, :selected_count, 5] = team_sign[selected]
        if ball_visible[frame]:
            offset = (players_m[frame, selected] - ball_m[frame]) * sign[frame]
            distance = np.linalg.norm(offset, axis=1)
            tokens[frame, :selected_count, 6:8] = np.clip(offset / 20.0, -3.0, 3.0)
            tokens[frame, :selected_count, 8] = np.exp(-distance / 2.0)
            tokens[frame, 0, 9] = 1.0

    visible_float = ball_visible.astype(np.float32)
    centered_ball = np.where(
        ball_visible[:, None], (ball - 0.5) * 2.0 * sign[:, None], 0.0
    )
    outside = ball_visible & ((ball < 0.0) | (ball > 1.0)).any(axis=1)
    since_ball, until_ball = _ball_gaps(ball_visible, blocks)
    counts = np.column_stack([
        (present & (teams[None, :] == team)).sum(axis=1) for team in (0, 1)
    ]).astype(np.float32)
    mean_speed = np.zeros((n, 2), dtype=np.float32)
    for team in (0, 1):
        team_present = present & (teams[None, :] == team)
        count = team_present.sum(axis=1)
        total = np.where(team_present, velocity_speed, 0.0).sum(axis=1)
        mean_speed[:, team] = np.divide(total, count, out=np.zeros(n, dtype=np.float32), where=count > 0) / 8.0

    glob = np.column_stack((
        visible_float,
        centered_ball,
        ball_velocity * sign[:, None] / 20.0,
        outside.astype(np.float32),
        since_ball,
        until_ball,
        counts / 11.0,
        mean_speed,
    )).astype(np.float32, copy=False)

    if n:
        X = np.asarray(data["X"], dtype=np.float32)
        if X.shape != (n, 16) or not np.isfinite(X).all():
            raise ValueError("X must be finite with shape (N, 16)")
        controls = np.column_stack((
            X[:, 0] / 30.0,
            X[:, 1] / 30.0,
            X[:, 2] / 10.0,
            X[:, 3],
            X[:, 7] / 34.0,
            X[:, 14] / 60.0,
            X[:, 15] / 60.0,
        ))
        glob = np.column_stack((glob, controls)).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, 16), dtype=np.float32)
        glob = np.zeros((0, len(GLOBAL_NAMES)), dtype=np.float32)

    return {
        "tokens": np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0),
        "mask": mask,
        "glob": np.nan_to_num(glob, nan=0.0, posinf=0.0, neginf=0.0),
        "X": X,
        "blocks": blocks,
    }


def mirror(tokens, glob, state):
    mirrored_tokens = np.array(tokens, copy=True)
    mirrored_glob = np.array(glob, copy=True)
    mirrored_state = np.asarray(state)
    mirrored_state = np.where(mirrored_state == 0, 1, np.where(mirrored_state == 1, 0, mirrored_state))
    mirrored_state = mirrored_state.astype(np.asarray(state).dtype, copy=False)
    mirrored_tokens[..., TOKEN_NEGATE] *= -1
    mirrored_glob[..., GLOBAL_NEGATE] *= -1
    for left, right in GLOBAL_SWAP:
        left_values = mirrored_glob[..., left].copy()
        mirrored_glob[..., left] = mirrored_glob[..., right]
        mirrored_glob[..., right] = left_values
    return mirrored_tokens, mirrored_glob, mirrored_state
