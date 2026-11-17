from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

import support as c

BASE_SEED = 20261113
DROP_FRACTION = 0.35
MIN_SPAN = 10
MAX_SPAN = 60


def _array_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if np.issubdtype(left.dtype, np.inexact) and np.issubdtype(right.dtype, np.inexact):
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _runs(mask, start, stop):
    local = np.asarray(mask[start:stop], dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, local, False].astype(np.int8)))
    return [(start + int(a), start + int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _half_groups(blocks, blocks25):
    if blocks.ndim != 2 or blocks.shape[1] != 2 or blocks25.shape != blocks.shape or len(blocks) < 2:
        raise ValueError('SkillCorner clean pack must retain two half timelines')
    if len(blocks) == 2:
        split = 1
    else:
        gaps = blocks25[1:, 0] - blocks25[:-1, 1]
        split = int(np.argmax(gaps)) + 1
        if gaps[split - 1] <= 0:
            split = len(blocks) // 2
    if split <= 0 or split >= len(blocks):
        raise ValueError('Cannot separate SkillCorner first and second half timelines')
    return (list(range(split)), list(range(split, len(blocks))))


def _span_lengths(target, maximum, rng):
    if target == 0:
        return []
    if target < MIN_SPAN or maximum < MIN_SPAN:
        raise ValueError('35% ball-drop target cannot be represented by 10..60-sample spans')
    count = math.ceil(target / maximum)
    shortest = target // count
    remainder = target % count
    lengths = [shortest + 1] * remainder + [shortest] * (count - remainder)
    if min(lengths) < MIN_SPAN or max(lengths) > min(MAX_SPAN, maximum):
        raise ValueError('Ball-drop target cannot be represented by valid contiguous spans')
    return sorted((int(length) for length in rng.permutation(lengths)), reverse=True)


def _drop_half(ball_visible, blocks, block_indices, seed):
    rng = np.random.default_rng(seed)
    visible_count = sum(int(np.count_nonzero(ball_visible[int(blocks[i, 0]):int(blocks[i, 1])]))
                        for i in block_indices)
    target = int(math.floor(visible_count * DROP_FRACTION + 0.5))
    available_runs = [run for i in block_indices
                      for run in _runs(ball_visible, int(blocks[i, 0]), int(blocks[i, 1]))]
    maximum = min(MAX_SPAN, max((stop - start for start, stop in available_runs), default=0))
    lengths = _span_lengths(target, maximum, rng)
    dropped = np.zeros(ball_visible.shape, dtype=bool)
    spans = []

    for length in lengths:
        options = []
        for block_index in block_indices:
            block_start, block_stop = (int(value) for value in blocks[block_index])
            free = ball_visible & ~dropped
            for run_start, run_stop in _runs(free, block_start, block_stop):
                first = run_start + int(run_start > block_start and dropped[run_start - 1])
                last = run_stop - int(run_stop < block_stop and dropped[run_stop])
                capacity = last - first
                if capacity >= length:
                    options.append((capacity, first, last, int(block_index)))
        if not options:
            raise ValueError('Cannot place exact ball-drop target in original visible spans')
        largest = max(option[0] for option in options)
        best = [option for option in options if option[0] == largest]
        capacity, first, last, block_index = best[int(rng.integers(len(best)))]
        if last - length == first:
            start = first
        else:
            start = first if rng.integers(2) == 0 else last - length
        stop = start + length
        dropped[start:stop] = True
        spans.append(dict(block=block_index, start_sample=int(start), stop_sample=int(stop), length=int(length)))

    if int(np.count_nonzero(dropped)) != target:
        raise ValueError('Ball-drop mask does not match requested visible-row count')
    return dropped, spans, visible_count


def _half_seed(match_id, half):
    return BASE_SEED + (2 * int(match_id)) + int(half)


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for value in arrays.values():
        if np.asarray(value).dtype.hasobject:
            raise ValueError('Object arrays are not allowed in stress cache packs')
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.part', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_feature_pack(result, sample_count):
    expected = {
        'player': (2, sample_count, 23),
        'ball': (2, sample_count, 5),
        'valid': (2, sample_count),
        'pp': (2, sample_count, 48, 10),
        'ppm': (2, sample_count, 48),
        'control': (2, sample_count, 7),
    }
    for key, shape in expected.items():
        if np.asarray(result[key]).shape != shape:
            raise ValueError(f'Invalid stress feature shape for {key}')
    for key in ('player', 'ball', 'control', 'pp'):
        if not np.isfinite(result[key]).all():
            raise ValueError(f'Nonfinite stress feature field: {key}')
    return {
        'player': np.asarray(result['player'], dtype=np.float32),
        'ball': np.asarray(result['ball'], dtype=np.float32),
        'valid': np.asarray(result['valid'], dtype=bool),
        'pp': np.asarray(result['pp'], dtype=np.float16),
        'ppm': np.asarray(result['ppm'], dtype=bool),
        'control': np.asarray(result['control'], dtype=np.float32),
        'blocks': np.asarray(result['blocks'], dtype=np.int64),
    }


def _source_code_hashes():
    wave1 = c.REPO / 'experiments' / 'exp_farras' / 'wave1' / 'data.py'
    return {
        'generator_sha256': c.digest(Path(__file__).resolve()),
        'wave1_feature_sha256': c.digest(wave1),
        'distance_helper_sha256': c.digest(c.PREVIOUS / 'prepare_skillcorner.py'),
        'feature_code_sha256': c.digest(c.PREVIOUS / 'features.py'),
    }


def _atomic_paths(name):
    return (
        c.CACHE / 'data' / f'{name}_stress.npz',
        c.CACHE / 'features' / f'{name}_stress.npz',
    )


def _validate_existing_entry(entry, name, source_sha256, pack_path, feature_path):
    if entry.get('source_sha256') != source_sha256:
        raise ValueError(f'Stress source changed: {name}')
    if entry.get('pack_path') != pack_path.relative_to(c.REPO).as_posix():
        raise ValueError(f'Stress pack path changed: {name}')
    if entry.get('feature_path') != feature_path.relative_to(c.REPO).as_posix():
        raise ValueError(f'Stress feature path changed: {name}')
    if not pack_path.is_file() or not feature_path.is_file():
        raise ValueError(f'Stress cache is incomplete: {name}')
    if entry.get('pack_sha256') != c.digest(pack_path):
        raise ValueError(f'Stress pack changed: {name}')
    if entry.get('feature_sha256') != c.digest(feature_path):
        raise ValueError(f'Stress feature pack changed: {name}')
    if entry.get('labels_preserved') is not True:
        raise ValueError(f'Stress label audit is invalid: {name}')


def _prepare_match(name, source_path, source_sha256, pack_path, feature_path, match_id,
                   wave1_data, distance_helper, feature_builder):
    data = c.base.load(name, 'clean')
    if data.get('_name') != name or data.get('_regime') != 'clean':
        raise ValueError(f'Unexpected clean source pack: {name}')
    original_arrays = {key: np.asarray(value) for key, value in data.items()
                       if not key.startswith('_') and isinstance(value, np.ndarray)}
    required = {'X', 'players', 'teams', 'ball', 'blocks', 'sampled', 'y', 'y25',
                'blocks25', 'n_frames', 'fps'}
    if not required.issubset(original_arrays):
        raise ValueError(f'Clean source pack lacks required arrays: {name}')
    blocks = np.asarray(original_arrays['blocks'], dtype=np.int64)
    blocks25 = np.asarray(original_arrays['blocks25'], dtype=np.int64)
    sample_count = len(original_arrays['ball'])
    cursor = 0
    for start, stop in blocks:
        if int(start) != cursor or int(stop) <= int(start) or int(stop) > sample_count:
            raise ValueError(f'Invalid sample blocks in clean source pack: {name}')
        cursor = int(stop)
    if cursor != sample_count or len(blocks25) != len(blocks):
        raise ValueError(f'Sample blocks do not preserve both timelines: {name}')
    if original_arrays['players'].shape[0] != sample_count or original_arrays['X'].shape != (sample_count, 16):
        raise ValueError(f'Player or feature rows do not match source timeline: {name}')

    half_groups = _half_groups(blocks, blocks25)
    original_ball = np.asarray(original_arrays['ball'])
    ball = original_ball.copy()
    original_visible = np.isfinite(original_ball).all(axis=1)
    dropped = np.zeros(sample_count, dtype=bool)
    half_audit = []
    for half_number, block_indices in enumerate(half_groups, start=1):
        seed = _half_seed(match_id, half_number)
        half_mask, spans, visible_count = _drop_half(original_visible, blocks, block_indices, seed)
        dropped |= half_mask
        first_block = int(block_indices[0])
        last_block = int(block_indices[-1])
        half_start = int(blocks[first_block, 0])
        half_stop = int(blocks[last_block, 1])
        half_rows = sum(int(blocks[index, 1] - blocks[index, 0]) for index in block_indices)
        after_visible = visible_count - int(np.count_nonzero(half_mask))
        ball[half_mask] = np.nan
        half_audit.append({
            'half': half_number,
            'seed': seed,
            'original_visible': visible_count,
            'dropped_visible': int(np.count_nonzero(half_mask)),
            'mask_count': int(np.count_nonzero(half_mask)),
            'after_visible': after_visible,
            'original_visible_fraction': float(visible_count / half_rows),
            'after_visible_fraction': float(after_visible / half_rows),
            'spans': spans,
            'sample_range': [half_start, half_stop],
        })

    distances = distance_helper._nearest_distances(original_arrays['players'], ball, original_arrays['teams'])
    X = wave1_data.features(original_arrays['players'], ball, distances, blocks.tolist())
    if X.shape != (sample_count, 16) or not np.isfinite(X).all():
        raise ValueError(f'Recomputed wave1 features are invalid: {name}')

    stress_arrays = dict(original_arrays)
    stress_arrays['ball'] = ball
    stress_arrays['X'] = np.asarray(X, dtype=np.float32)
    if not np.array_equal(stress_arrays['y'], original_arrays['y']):
        raise ValueError(f'Sampled labels changed: {name}')
    if not np.array_equal(stress_arrays['y25'], original_arrays['y25']):
        raise ValueError(f'25fps labels changed: {name}')
    for key in ('players', 'teams', 'sampled', 'blocks', 'blocks25', 'n_frames', 'fps'):
        if not _array_equal(stress_arrays[key], original_arrays[key]):
            raise ValueError(f'Stress changed timeline or player data: {name} ({key})')
    if set(stress_arrays) != set(original_arrays):
        raise ValueError(f'Stress pack fields changed: {name}')
    for key in stress_arrays.keys() - {'X', 'ball'}:
        if not _array_equal(stress_arrays[key], original_arrays[key]):
            raise ValueError(f'Stress changed source field: {name} ({key})')

    feature_result = feature_builder.build(c.base.observations({**data, 'ball': ball, 'X': stress_arrays['X']}))
    feature_arrays = _validate_feature_pack(feature_result, sample_count)
    _atomic_npz(pack_path, stress_arrays)
    _atomic_npz(feature_path, feature_arrays)
    if c.digest(source_path) != source_sha256:
        raise ValueError(f'Clean source pack changed during stress preparation: {name}')

    total_visible = int(np.count_nonzero(original_visible))
    dropped_visible = int(np.count_nonzero(dropped))
    after_visible = total_visible - dropped_visible
    total_rows = sample_count
    return {
        'match_id': int(match_id),
        'source_path': source_path.relative_to(c.REPO).as_posix(),
        'source_sha256': source_sha256,
        'pack_path': pack_path.relative_to(c.REPO).as_posix(),
        'pack_sha256': c.digest(pack_path),
        'feature_path': feature_path.relative_to(c.REPO).as_posix(),
        'feature_sha256': c.digest(feature_path),
        'labels_preserved': True,
        'original_visible': total_visible,
        'dropped_visible': dropped_visible,
        'after_visible': after_visible,
        'original_visible_fraction': float(total_visible / total_rows),
        'after_visible_fraction': float(after_visible / total_rows),
        'dropped_fraction_of_original_visible': float(dropped_visible / total_visible) if total_visible else 0.0,
        'halves': half_audit,
    }


def prepare():
    selection_path = c.OUT / 'selection.json'
    if not selection_path.is_file():
        raise FileNotFoundError('Freeze wave5 selection before stress preparation')
    selection_sha256 = c.digest(selection_path)
    code_hashes = _source_code_hashes()
    header = {
        'schema_version': 1,
        'seed': BASE_SEED,
        'drop_fraction': DROP_FRACTION,
        'span_samples': [MIN_SPAN, MAX_SPAN],
        'selection_sha256': selection_sha256,
        **code_hashes,
    }
    audit_path = c.CACHE / 'stress_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if not isinstance(audit, dict) or not isinstance(audit.get('packs'), dict):
            raise ValueError('Existing stress audit has invalid schema')
        for key, value in header.items():
            if audit.get(key) != value:
                raise ValueError(f'Stress provenance changed: {key}')
        if set(audit['packs']) - {f'{name}_stress' for name in c.SC_HOLD}:
            raise ValueError('Existing stress audit contains non-holdout packs')
    else:
        audit = {**header, 'packs': {}}

    if len(c.SC_HOLD) != 6 or any(not name.startswith('skillcorner_') for name in c.SC_HOLD):
        raise ValueError('Stress preparation requires exactly six SkillCorner holdout matches')

    wave1_path = c.REPO / 'experiments' / 'exp_farras' / 'wave1' / 'data.py'
    feature_path = c.PREVIOUS / 'features.py'
    helper_path = c.PREVIOUS / 'prepare_skillcorner.py'
    wave1_data = c.import_path('wave5_wave1_data', wave1_path)
    distance_helper = c.import_path('wave5_skillcorner_prepare', helper_path)
    feature_builder = c.import_path('wave5_wave4_features', feature_path)

    for name in c.SC_HOLD:
        match_id = int(name.rpartition('_')[2])
        source_path = c.base.pack_path(name, 'clean')
        source_sha256 = c.digest(source_path)
        pack_path, feature_output_path = _atomic_paths(name)
        key = f'{name}_stress'
        entry = audit['packs'].get(key)
        if entry is not None:
            _validate_existing_entry(entry, name, source_sha256, pack_path, feature_output_path)
            if c.digest(selection_path) != selection_sha256:
                raise ValueError('Wave5 selection changed during stress preparation')
            continue
        if pack_path.exists() or feature_output_path.exists():
            raise ValueError(f'Untracked stress cache exists; refusing overwrite: {name}')
        result = _prepare_match(name, source_path, source_sha256, pack_path, feature_output_path,
                                match_id, wave1_data, distance_helper, feature_builder)
        audit['packs'][key] = result
        c.write_json(audit_path, audit)
        if c.digest(selection_path) != selection_sha256:
            raise ValueError('Wave5 selection changed during stress preparation')

    if c.digest(selection_path) != selection_sha256:
        raise ValueError('Wave5 selection changed during stress preparation')
    if set(audit['packs']) != {f'{name}_stress' for name in c.SC_HOLD}:
        raise ValueError('Stress preparation did not produce all six holdout packs')
    c.write_json(audit_path, audit)
    return audit


if __name__ == '__main__':
    manifest = prepare()
    print('STRESS READY', len(manifest['packs']), flush=True)
