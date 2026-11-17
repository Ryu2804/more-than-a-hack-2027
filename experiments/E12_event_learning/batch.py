from __future__ import annotations

import numpy as np

import bridge as b


_FAMILIES = ('balanced', 'focal', 'boundary', 'ranking', 'ema', 'contact')


def attach(data, control):
    features = data['_features']
    player = np.asarray(features['player'])
    if player.ndim != 3 or player.shape[0] != 2 or player.shape[2] != 23:
        raise ValueError('player features must have shape (2, N, 23)')
    values = np.asarray(control)
    if values.shape != (2, player.shape[1], 7) or not np.isfinite(values).all():
        raise ValueError('control features must be finite with shape (2, N, 7)')
    return {**data, '_features': {**features, 'control': values}}


def make_batch(records, grouped, mean, scale, config, rng, family):
    if family not in _FAMILIES:
        raise ValueError(f'Unknown training family: {family!r}')
    batch = int(config['batch'])
    crop = int(config['crop'])
    contact = family == 'contact'
    boundary = family == 'boundary'
    x_batch = np.zeros((batch, 36 if contact else 29, crop), dtype=np.float32)
    pp_batch = np.empty((batch, crop, 48, 10), dtype=np.float16)
    ppm_batch = np.empty((batch, crop, 48), dtype=np.bool_)
    y_batch = np.empty((batch, crop), dtype=np.int64)
    view_batch = np.empty(batch, dtype=np.int64)
    weight_batch = np.empty((batch, crop), dtype=np.float32)
    boundary_target = np.empty((batch, crop), dtype=np.float32) if boundary else None
    boundary_valid = np.empty((batch, crop), dtype=np.bool_) if boundary else None
    for batch_index in range(batch):
        record, start, view = b.old_sampling.draw(records, grouped, rng, crop, event_probability=.5)
        end = start + crop
        mask = b.original._random_mask(crop, float(config['mask_full']), rng)
        player = record['player'][view, start:end].astype(np.float32)
        player = np.nan_to_num((player - mean) / scale, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if float(config['noise']):
            player += float(config['noise']) * rng.standard_normal(player.shape).astype(np.float32)
        x_batch[batch_index, :23] = player.T
        x_batch[batch_index, 23] = record['valid'][view, start:end]
        ball = record['ball'][view, start:end].astype(np.float32)
        ball = np.nan_to_num(ball, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        ball[mask] = 0.0
        x_batch[batch_index, 24:29] = ball.T
        if contact:
            control = record['control'][view, start:end].astype(np.float32)
            control = np.nan_to_num(control, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            control[mask] = 0.0
            x_batch[batch_index, 29:36] = control.T
        per_player = record['pp'][view, start:end].copy()
        per_player[mask, :, 6:9] = 0.0
        pp_batch[batch_index] = per_player
        ppm_batch[batch_index] = record['ppm'][view, start:end]
        y_batch[batch_index] = record['y'][start:end]
        view_batch[batch_index] = view
        weight_batch[batch_index] = record['weights'][start:end]
        if boundary:
            boundary_target[batch_index] = record['boundary_target'][start:end]
            boundary_valid[batch_index] = record['boundary_valid'][start:end]
    return x_batch, pp_batch, ppm_batch, y_batch, view_batch, weight_batch, boundary_target, boundary_valid
