"""Half-local contact and touch-memory features from degraded 25 fps observations."""

import numpy as np

FPS = 25
TOUCH_RADIUS_METERS = 0.8
TOUCH_MIN_FRAMES = 30
OUT_MIN_FRAMES = 15
SLOW_MIN_FRAMES = 25
DEAD_MERGE_FRAMES = 150


def half_blocks(half):
    half = np.asarray(half)
    if half.ndim != 1:
        raise ValueError('half must be one-dimensional')
    starts = np.flatnonzero(np.r_[True, half[1:] != half[:-1]]) if len(half) else []
    ends = np.r_[starts[1:], len(half)]
    return [(int(a), int(z)) for a, z in zip(starts, ends) if half[a] > 0]


def run_mask(flags, blocks, minimum):
    """Mark whole qualifying runs; adjacent halves never join."""
    flags = np.asarray(flags, dtype=bool)
    output = np.zeros(len(flags), dtype=bool)
    for start, stop in blocks:
        edges = np.diff(np.r_[False, flags[start:stop], False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if right - left >= minimum:
                output[start + left:start + right] = True
    return output


def player_speeds(players, half):
    """Positions are metres; provider speed columns are not used."""
    xy = np.asarray(players, dtype=np.float64)[..., :2]
    speed = np.full(xy.shape[:2], np.nan)
    for start, stop in half_blocks(half):
        local = xy[start:stop]
        valid = np.isfinite(local[1:]).all(axis=2) & np.isfinite(local[:-1]).all(axis=2)
        with np.errstate(invalid='ignore', over='ignore'):
            derived = np.linalg.norm(np.diff(local, axis=0), axis=2) * FPS
        speed[start + 1:stop] = np.where(valid & np.isfinite(derived), derived, np.nan)
    return speed


def source_geometry(players, ball, teams, half):
    xy = np.asarray(players, dtype=np.float64)[..., :2]
    ball = np.asarray(ball, dtype=np.float64)
    teams = np.asarray(teams)
    n = len(ball)
    if xy.ndim != 3 or xy.shape[0] != n or xy.shape[2] != 2 or ball.shape != (n, 2):
        raise ValueError('Expected player XY and ball XY on one source timeline')
    if teams.shape != (xy.shape[1],) or np.asarray(half).shape != (n,):
        raise ValueError('Player metadata or half timeline mismatch')
    distance = np.full((n, 2), np.inf)
    nearest_distance = np.full(n, np.inf)
    nearest_team = np.full(n, -1, dtype=np.int8)
    nearest_speed = np.full(n, np.nan)
    slots = np.flatnonzero(np.isin(teams, (0, 1)))
    if len(slots):
        speed = player_speeds(xy, half)
        for start in range(0, n, 8192):
            stop = min(start + 8192, n)
            known = xy[start:stop, slots]
            with np.errstate(invalid='ignore', over='ignore'):
                distances = np.linalg.norm(known - ball[start:stop, None], axis=2)
            valid = (np.isfinite(known).all(axis=2)
                     & np.isfinite(ball[start:stop]).all(axis=1)[:, None]
                     & np.isfinite(distances))
            distances = np.where(valid, distances, np.inf)
            for team in (0, 1):
                selected = teams[slots] == team
                if selected.any():
                    distance[start:stop, team] = distances[:, selected].min(axis=1)
            index = distances.argmin(axis=1)
            rows = np.arange(stop - start)
            closest = distances[rows, index]
            answered = np.isfinite(closest)
            selected_slots = slots[index]
            nearest_distance[start:stop] = closest
            nearest_team[start:stop] = np.where(answered, teams[selected_slots], -1)
            nearest_speed[start:stop] = np.where(answered, speed[start:stop][rows, selected_slots], np.nan)
    return dict(distance=distance, nearest_distance=nearest_distance,
                nearest_team=nearest_team, nearest_speed=nearest_speed)


def ball_impulse(ball, half):
    """Jeremy's dv is a velocity difference, not acceleration in m/s squared."""
    ball = np.asarray(ball, dtype=np.float64)
    dv = np.full(len(ball), np.nan)
    for start, stop in half_blocks(half):
        local = ball[start:stop]
        if len(local) < 3:
            continue
        visible = np.isfinite(local).all(axis=1)
        valid = visible[2:] & visible[1:-1] & visible[:-2]
        with np.errstate(invalid='ignore', over='ignore'):
            values = np.linalg.norm(local[2:] - 2 * local[1:-1] + local[:-2], axis=1) * FPS
        dv[start + 2:stop] = np.where(valid & np.isfinite(values), values, np.nan)
    return dv


def contact_features(ball, half, sampled, geometry):
    sampled = np.asarray(sampled, dtype=np.int64)
    distance = np.asarray(geometry['distance'])
    nearest_team = np.asarray(geometry['nearest_team'])
    impulse = ball_impulse(ball, half)
    output = np.zeros((len(sampled), 4), dtype=np.float32)
    output[:, :2] = 3
    peak = np.full(len(sampled), -np.inf)
    block_start = np.full(len(ball), -1, dtype=np.int64)
    for start, stop in half_blocks(half):
        block_start[start:stop] = start
    for offset in range(4, -1, -1):
        source = sampled - offset
        valid = (block_start[sampled] >= 0) & (source >= block_start[sampled])
        rows = np.flatnonzero(valid)
        indices = source[rows]
        output[rows, :2] = np.minimum(output[rows, :2], np.minimum(distance[indices] / 10, 3))
        values = impulse[indices]
        improved = np.isfinite(values) & (values > peak[rows])
        winners = rows[improved]
        peak[winners] = values[improved]
        team = nearest_team[indices[improved]]
        output[winners, 3] = np.where(team == 0, 1, np.where(team == 1, -1, 0))
    finite = np.isfinite(peak)
    output[finite, 2] = np.minimum(peak[finite] / 20, 3)
    return output


def carry_team(touch_team, half, minimum_run=TOUCH_MIN_FRAMES):
    """Absorb short carried-team runs into their predecessor, never future-fill."""
    touch_team = np.asarray(touch_team)
    held = np.full(len(touch_team), 2, dtype=np.int8)
    for start, stop in half_blocks(half):
        local = touch_team[start:stop]
        touches = np.flatnonzero(np.isin(local, (0, 1)))
        if not len(touches):
            continue
        previous = np.searchsorted(touches, np.arange(stop - start), side='right') - 1
        answered = previous >= 0
        segment = held[start:stop]
        segment[answered] = local[touches[previous[answered]]]
        starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
        ends = np.r_[starts[1:], len(segment)]
        for left, right in zip(starts, ends):
            if segment[left] != 2 and right - left < minimum_run:
                segment[left:right] = segment[left - 1] if left else 2
    return held


def contact_recency(touch_team, half):
    touches = np.isin(touch_team, (0, 1))
    recency = np.zeros(len(touches), dtype=np.float32)
    for start, stop in half_blocks(half):
        frame = np.arange(stop - start)
        last = np.maximum.accumulate(np.where(touches[start:stop], frame, -1))
        known = last >= 0
        local = recency[start:stop]
        local[known] = np.exp(-(frame[known] - last[known]) / (FPS * 10))
    return recency


def dead_cues(ball, half, pitch, geometry, merge_frames=DEAD_MERGE_FRAMES):
    ball = np.asarray(ball)
    visible = np.isfinite(ball).all(axis=1)
    outside = visible & ((ball < 0).any(axis=1) | (ball > np.asarray(pitch)).any(axis=1))
    speed = np.asarray(geometry['nearest_speed'])
    slow = (visible & np.isfinite(geometry['nearest_distance'])
            & np.isfinite(speed) & (speed < 2))
    blocks = half_blocks(half)
    cue = run_mask(outside, blocks, OUT_MIN_FRAMES) | run_mask(slow, blocks, SLOW_MIN_FRAMES)
    return run_mask(cue, blocks, merge_frames)


def mirror_view(control):
    control = np.asarray(control)
    if control.ndim != 2 or control.shape[1] != 7:
        raise ValueError('Expected seven control channels')
    mirrored = control[:, [1, 0, 2, 3, 4, 5, 6]].copy()
    mirrored[:, [3, 4]] *= -1
    return mirrored


def assert_regeneration(regenerated, prepared):
    """Reject any cached timeline or sampled-observation mismatch before features."""
    if regenerated.labels is not None or 'y' in prepared or 'y25' in prepared:
        raise ValueError('Feature construction requires label-free inputs')
    pitch = np.asarray(regenerated.pitch, dtype=np.float32)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError('Invalid pitch dimensions')
    blocks25 = half_blocks(regenerated.half)
    sampled_parts = [np.arange(a, z, 5, dtype=np.int64) for a, z in blocks25]
    sampled = np.concatenate(sampled_parts) if sampled_parts else np.empty(0, dtype=np.int64)
    lengths = np.asarray([len(part) for part in sampled_parts], dtype=np.int64)
    ends = np.cumsum(lengths)
    blocks = np.column_stack([ends - lengths, ends]).reshape(-1, 2)
    exact = dict(sampled=sampled, teams=np.asarray(regenerated.teams),
                 blocks=blocks, blocks25=np.asarray(blocks25, dtype=np.int64).reshape(-1, 2),
                 n_frames=np.asarray(len(regenerated.ball)), fps=np.asarray(5))
    for key, expected in exact.items():
        observed = np.asarray(prepared[key])
        if observed.shape != expected.shape or not np.array_equal(observed, expected, equal_nan=True):
            raise ValueError(f'Regeneration mismatch: {key}')
    observations = dict(players=np.asarray(regenerated.players)[sampled, :, :2] / pitch,
                        ball=np.asarray(regenerated.ball)[sampled] / pitch)
    for key, expected in observations.items():
        observed = np.asarray(prepared[key])
        if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f'Regeneration mismatch: {key}')
    return sampled


def source_features(players, ball, teams, half, pitch, sampled):
    geometry = source_geometry(players, ball, teams, half)
    touch_team = np.where(geometry['nearest_distance'] < TOUCH_RADIUS_METERS,
                          geometry['nearest_team'], -1)
    held = carry_team(touch_team, half)
    recency = contact_recency(touch_team, half)
    cue = dead_cues(ball, half, pitch, geometry)
    native = held.copy()
    native[cue] = 2
    control = np.empty((len(sampled), 7), dtype=np.float32)
    control[:, :4] = contact_features(ball, half, sampled, geometry)
    control[:, 4] = np.where(held[sampled] == 0, 1, np.where(held[sampled] == 1, -1, 0))
    control[:, 5] = recency[sampled]
    control[:, 6] = cue[sampled]
    audit = dict(source_fps=FPS, sampled_frames=int(len(sampled)), source_frames=int(len(ball)),
                 halves=int(len(half_blocks(half))),
                 metadata_known_slots=int(np.isin(teams, (0, 1)).sum()),
                 metadata_ignored_slots=int((~np.isin(teams, (0, 1))).sum()),
                 visible_ball_frames=int(np.isfinite(ball).all(axis=1).sum()),
                 contact_frames=int(np.isin(touch_team, (0, 1)).sum()),
                 held_home_frames=int((held == 0).sum()), held_away_frames=int((held == 1).sum()),
                 held_unknown_frames=int((held == 2).sum()), dead_cue_frames=int(cue.sum()))
    return dict(control=np.stack([control, mirror_view(control)]).astype(np.float16),
                native_codes=native[sampled].astype(np.int8), audit=audit)


def build(clean, prepared, seed):
    if clean.labels is not None or 'y' in prepared or 'y25' in prepared:
        raise ValueError('Feature construction requires label-free inputs')
    import bridge as b

    regenerated = b.reference.degrade(clean, seed=seed)
    sampled = assert_regeneration(regenerated, prepared)
    output = source_features(regenerated.players, regenerated.ball, regenerated.teams,
                             regenerated.half, regenerated.pitch, sampled)
    output['audit'].update(seed=int(seed), regenerated_observations_match=True, label_access=False)
    return output
