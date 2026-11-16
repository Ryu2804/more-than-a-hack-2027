import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / 'wave3'
OUT = HERE / 'out'
CACHE = HERE / 'cache'
IDSSE_DEV = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY')
IDSSE_HOLD = ('J03WQQ', 'J03WR9')
SC_IDS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691,
          1996435, 1996436, 2006229, 2006363, 2007448, 2007721, 2010085, 2011166,
          2013725, 2015213, 2016236, 2017461)
SC_DEV = tuple(f'skillcorner_{number}' for number in SC_IDS[:14])
SC_HOLD = tuple(f'skillcorner_{number}' for number in SC_IDS[14:])
DEV = (*IDSSE_DEV, *SC_DEV)
HOLD = (*IDSSE_HOLD, *SC_HOLD)
FOLD = {name: index % 6 for names in (sorted(IDSSE_DEV), sorted(SC_DEV)) for index, name in enumerate(names)}


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pm = import_path('wave4_official_metrics', REPO / 'evaluation/possession_metrics.py')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source(name):
    return 'skillcorner' if name.startswith('skillcorner_') else ('idsse' if name.startswith('J03') else 'metrica')


def pack_path(name, regime):
    parent = CACHE if name.startswith('skillcorner_') else PREVIOUS / 'cache'
    return parent / 'data' / f'{name}_{regime}.npz'


def load(name, regime, with_features=False, light=False):
    with np.load(pack_path(name, regime), allow_pickle=False) as archive:
        keys = ('y', 'y25', 'blocks', 'blocks25', 'n_frames', 'fps', 'sampled') if light else archive.files
        data = {key: archive[key] for key in keys}
    data.update(_name=name, _source=source(name), _regime=regime)
    if with_features:
        with np.load(CACHE / 'features' / f'{name}_{regime}.npz', allow_pickle=False) as archive:
            data['_features'] = {key: archive[key] for key in archive.files}
    return data


def training(names):
    return [load(name, regime, with_features=True, light=True) for name in names
            for regime in (('r0', 'r1', 'r2') if name.startswith('J03') else ('clean',))]


def observations(data):
    return {key: value for key, value in data.items() if key not in ('y', 'y25')}


def decoder_observations(data):
    result = observations(data)
    players, ball, teams = data['players'], data['ball'], data['teams']
    distances = np.full((len(ball), 2), np.nan, dtype=np.float32)
    for start in range(0, len(ball), 4096):
        stop = min(start + 4096, len(ball))
        delta = (players[start:stop] - ball[start:stop, None]) * [105.3, 68.0]
        distance = np.linalg.norm(delta, axis=2)
        for team in (0, 1):
            active = (teams == team)[None, :] & np.isfinite(distance)
            nearest = np.min(np.where(active, distance, np.inf), axis=1)
            distances[start:stop, team] = np.where(np.isfinite(nearest), nearest, np.nan)
    result['distances'] = distances
    return result


def training_stats(names):
    by_source = {}
    for name in names:
        regime = 'r0' if name.startswith('J03') else 'clean'
        data = load(name, regime, light=True)
        y = data['y25']
        valid = 0
        spells = 0
        dead = 0
        for start, stop in data['blocks25']:
            part = y[start:stop]
            valid += int(np.sum(part >= 0))
            dead += int(np.sum(part == 2))
            for state in (0, 1):
                active = part == state
                spells += int(np.sum(active & ~np.r_[False, active[:-1]]))
        if not valid:
            raise ValueError('No labeled training duration: ' + name)
        row = dict(rate=spells / (valid / 25 / 60), dead_share=dead / valid)
        by_source.setdefault(source(name), []).append(row)
    if not by_source:
        raise ValueError('Training statistics need matches')
    return {key: float(np.mean([np.mean([row[key] for row in rows]) for rows in by_source.values()]))
            for key in ('rate', 'dead_share')}


def expand(codes, data):
    output = np.full(int(data['n_frames']), 3, dtype=np.int8)
    for (start, stop), (begin, end) in zip(data['blocks'], data['blocks25']):
        output[begin:end] = np.repeat(codes[start:stop], 5)[:end - begin]
    return output


def metrics(codes, data):
    prediction = expand(codes, data)
    result = pm.evaluate(np.where(data['y25'] < 0, 3, data['y25']).astype(np.int8), prediction, bootstrap_n=0)
    row = {key: result[key] for key in ('spell_f1_mean', 'macro_f1', 'accuracy', 'n_gt_spells', 'n_pred_spells', 'mean_spell_duration_s')}
    row['count_error_ratio'] = float((row['n_pred_spells'] - row['n_gt_spells']) / max(1, row['n_gt_spells']))
    return row


def dev_cases():
    return [(name, 'r0' if name.startswith('J03') else 'clean', f'fold{FOLD[name]}') for name in DEV] + [
        (f'metrica{number}', regime, 'final') for number in (1, 2) for regime in ('clean', 'r0')]


def holdout_cases():
    return [(name, 'r0' if name.startswith('J03') else 'clean') for name in HOLD] + [
        ('metrica3', 'clean'), ('metrica3', 'r0')]
