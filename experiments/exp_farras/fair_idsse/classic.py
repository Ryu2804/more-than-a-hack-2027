"""Six-match HGB and a half-local 5 fps touch-hold adaptation, not exact Rasya."""

import numpy as np

from bridge import probability_result, tree

FPS = 5
PITCH_METERS = np.array([105.3, 68.0])
TOUCH_RADIUS_METERS = 0.8
TOUCH_MIN_SAMPLES = 6
OUT_MIN_SAMPLES = 3
SLOW_MIN_SAMPLES = 5
SLOW_SPEED_MPS = 2.0
DEAD_MERGE_SAMPLES = 0


def pool_training(datasets):
    datasets = tuple(datasets)
    names = tuple(data['_name'] for data in datasets)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('HGB requires six unique training matches')
    features, labels, blocks = [], [], []
    offset = 0
    for data in datasets:
        features.append(data['X'])
        labels.append(data['y'])
        blocks.extend((int(start) + offset, int(stop) + offset)
                      for start, stop in data['blocks'])
        offset += len(data['y'])
    return dict(X=np.concatenate(features), y=np.concatenate(labels),
                blocks=np.asarray(blocks, dtype=np.int64), training_names=names)


def tree_fit(datasets):
    pooled = pool_training(datasets)
    fitted = tree.fit_emitter(tree.symmetrize(pooled), kind='hist')
    fitted['training_names'] = pooled['training_names']
    return fitted


def tree_predict(fitted, observations):
    return probability_result(tree.probabilities(fitted, observations))


def player_speeds(players, blocks):
    """Backward differences in m/s; half starts and missing endpoints stay NaN."""
    positions = np.asarray(players, dtype=np.float64)
    speed = np.full(positions.shape[:2], np.nan)
    for start, stop in blocks:
        start, stop = int(start), int(stop)
        if stop - start < 2:
            continue
        local = positions[start:stop]
        valid = np.isfinite(local[1:]).all(axis=2) & np.isfinite(local[:-1]).all(axis=2)
        delta = np.diff(local, axis=0) * PITCH_METERS
        with np.errstate(invalid='ignore', over='ignore'):
            derived = np.linalg.norm(delta, axis=2) * FPS
        speed[start + 1:stop] = np.where(valid & np.isfinite(derived), derived, np.nan)
    return speed


def visible_ball(observations):
    visibility = np.asarray(observations['X'])[:, 3]
    return (np.isfinite(visibility) & (visibility > 0.5)
            & np.isfinite(observations['ball']).all(axis=1))


def nearest_player(observations):
    """Only metadata-labelled players can provide touch or slow-speed evidence."""
    players = np.asarray(observations['players'], dtype=np.float64)
    ball = np.asarray(observations['ball'], dtype=np.float64)
    metadata = np.asarray(observations['teams'])
    slots = np.flatnonzero((metadata == 0) | (metadata == 1))
    n = len(ball)
    distance = np.full(n, np.inf)
    team = np.full(n, -1, dtype=np.int8)
    speed = np.full(n, np.nan)
    if not len(slots):
        return distance, team, speed
    known = players[:, slots]
    with np.errstate(invalid='ignore', over='ignore'):
        distances = np.linalg.norm((known - ball[:, None]) * PITCH_METERS, axis=2)
    valid = (np.isfinite(known).all(axis=2) & np.isfinite(distances)
             & visible_ball(observations)[:, None])
    distances = np.where(valid, distances, np.inf)
    index = np.argmin(distances, axis=1)
    rows = np.arange(n)
    distance = distances[rows, index]
    answered = np.isfinite(distance)
    selected = slots[index]
    team[answered] = metadata[selected[answered]]
    derived = player_speeds(players, observations['blocks'])
    speed[answered] = derived[rows[answered], selected[answered]]
    return distance, team, speed


def run_mask(flags, blocks, minimum):
    """Mark whole qualifying runs, without joining adjacent halves."""
    flags = np.asarray(flags, dtype=bool)
    mask = np.zeros(len(flags), dtype=bool)
    for start, stop in blocks:
        start, stop = int(start), int(stop)
        edges = np.diff(np.r_[False, flags[start:stop], False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if right - left >= minimum:
                mask[start + left:start + right] = True
    return mask


def carry_team(touch_team, blocks, minimum_run=TOUCH_MIN_SAMPLES):
    """Carry touches, then absorb short held-team runs into their predecessor.

    Leading non-touch stays D; no future team is backfilled before a first touch.
    """
    touch_team = np.asarray(touch_team)
    held = np.full(len(touch_team), 2, dtype=np.int8)
    for start, stop in blocks:
        start, stop = int(start), int(stop)
        local = touch_team[start:stop]
        touches = np.flatnonzero((local == 0) | (local == 1))
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
                segment[left:right] = segment[left - 1] if left > 0 else 2
    return held


def dead_cues(observations, nearest=None):
    if nearest is None:
        nearest = nearest_player(observations)
    distance, _, speed = nearest
    ball = np.asarray(observations['ball'])
    visible = visible_ball(observations)
    outside = visible & ((ball < 0).any(axis=1) | (ball > 1).any(axis=1))
    slow = visible & np.isfinite(distance) & np.isfinite(speed) & (speed < SLOW_SPEED_MPS)
    blocks = observations['blocks']
    return (run_mask(outside, blocks, OUT_MIN_SAMPLES)
            | run_mask(slow, blocks, SLOW_MIN_SAMPLES))


def touch_predict(observations):
    nearest = nearest_player(observations)
    distance, team, _ = nearest
    touch_team = np.where(distance < TOUCH_RADIUS_METERS, team, -1)
    held = carry_team(touch_team, observations['blocks'])
    # Dead merging is disabled: a qualifying cue is never erased by its duration.
    held[dead_cues(observations, nearest)] = 2
    return held
