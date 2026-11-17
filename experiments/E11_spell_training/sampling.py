from __future__ import annotations

import numpy as np

import bridge as b
import weights


def prepare(records, family, crop):
    if family not in ('spell_balanced_tcn', 'event_tcn', 'dual_scale_tcn'):
        raise ValueError(f'Unknown training family: {family!r}')
    prepared = []
    for record in records:
        labels = b.training._labels(record['y'], (1,))
        blocks = b.original._normalize_blocks(record['blocks'], len(labels))
        eligible = [(start, end) for start, end in blocks if end - start >= crop]
        if not eligible:
            raise ValueError(f"{record['source']}:{record['name']} has no block at least crop frames long")
        full_weights = (
            weights.spell_weights(labels, blocks)
            if family == 'spell_balanced_tcn'
            else b.training.short_spell_weights(labels, blocks)
        )
        events = []
        if family == 'event_tcn':
            events = [
                event for event in weights._complete_runs(labels, eligible)
                if event[3] - event[2] < 20
            ]
        prepared.append({
            **record,
            'blocks': eligible,
            'weights': full_weights,
            'short_events': np.asarray(events, dtype=np.int64).reshape(-1, 4),
        })
    return prepared


def draw(records, grouped, rng, crop, event_probability):
    record, start, view = b.original._sample_record(records, grouped, rng, crop)
    if event_probability == 0:
        return record, start, view
    if rng.random() < event_probability and len(record['short_events']):
        events = record['short_events']
        block_start, block_end, run_start, run_end = events[int(rng.integers(len(events)))]
        jitter = int(rng.integers(-(crop // 4), crop // 4 + 1))
        center = (int(run_start) + int(run_end)) // 2
        start = int(np.clip(center - crop // 2 + jitter, block_start, block_end - crop))
    return record, start, view


def make_batch(records, grouped, mean, scale, config, rng, family):
    batch = int(config['batch'])
    crop = int(config['crop'])
    x_batch = np.zeros((batch, 29, crop), dtype=np.float32)
    pp_batch = np.empty((batch, crop, 48, 10), dtype=np.float16)
    ppm_batch = np.empty((batch, crop, 48), dtype=np.bool_)
    y_batch = np.empty((batch, crop), dtype=np.int64)
    view_batch = np.empty(batch, dtype=np.int64)
    weight_batch = np.empty((batch, crop), dtype=np.float32)
    probability = 0.5 if family == 'event_tcn' else 0.0
    for batch_index in range(batch):
        record, start, view = draw(records, grouped, rng, crop, probability)
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
        per_player = record['pp'][view, start:end].copy()
        per_player[mask, :, 6:9] = 0.0
        pp_batch[batch_index] = per_player
        ppm_batch[batch_index] = record['ppm'][view, start:end]
        y_batch[batch_index] = record['y'][start:end]
        view_batch[batch_index] = view
        weight_batch[batch_index] = record['weights'][start:end]
    return x_batch, pp_batch, ppm_batch, y_batch, view_batch, weight_batch
