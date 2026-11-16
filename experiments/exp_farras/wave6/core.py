import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'wave5'
sys.path.insert(0, str(PREVIOUS))
import support as c

previous = c.import_path('wave6_frozen_wave5_runner', PREVIOUS / 'run.py')
decoder = previous.decoder
SPEC = previous.settings(decoder, 'dead45')
OUT = HERE / 'out'


def result(probability):
    values = np.array(probability, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Joint probabilities must be finite nonnegative (N, 3)')
    if np.any(values.sum(axis=1) <= 0):
        raise ValueError('Probability rows need positive mass')
    values /= values.sum(axis=1, keepdims=True)
    live = values[:, 0] + values[:, 1]
    team = np.divide(values[:, 0], live, out=np.full(len(values), 0.5), where=live > 0)
    return dict(probability=values, p_dead=values[:, 2], p_team=team,
        boundary=np.full(len(values), 0.5), has_boundary=False)


def from_heads(p_dead, p_team):
    dead, team = np.asarray(p_dead), np.asarray(p_team)
    if dead.ndim != 1 or team.shape != dead.shape:
        raise ValueError('Heads must be equal-length vectors')
    if not np.isfinite(dead).all() or not np.isfinite(team).all() or np.any((dead < 0) | (dead > 1)) or np.any((team < 0) | (team > 1)):
        raise ValueError('Heads must contain probabilities')
    output = result(np.column_stack(((1 - dead) * team, (1 - dead) * (1 - team), dead)))
    output['p_dead'], output['p_team'] = dead.copy(), team.copy()
    return output


def mix(first, second, weight_second):
    left, right = first['probability'], second['probability']
    weight = np.asarray(weight_second, dtype=np.float64)
    if left.shape != right.shape or weight.shape not in ((), (len(left),)):
        raise ValueError('Mixture shape mismatch')
    if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
        raise ValueError('Mixture weights must be probabilities')
    if weight.ndim:
        weight = weight[:, None]
    return result((1 - weight) * left + weight * right)


def mean_by_half(values, blocks, window):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or window < 1 or window % 2 != 1:
        raise ValueError('Finite vector and positive odd window required')
    ranges = decoder._normalize_blocks(blocks, len(values))
    output = np.empty_like(values)
    radius = window // 2
    for start, end in ranges:
        prefix = np.r_[0.0, np.cumsum(values[start:end])]
        index = np.arange(end - start)
        left = np.maximum(index - radius, 0)
        right = np.minimum(index + radius + 1, end - start)
        output[start:end] = (prefix[right] - prefix[left]) / (right - left)
    return output


def decode(prediction, observations, stats, dead=None, transition_weights=None, postprocess=None):
    if dead is None and transition_weights is None and postprocess is None:
        return decoder.decode(prediction, observations, SPEC, stats)['codes']
    n = len(observations['X'])
    blocks = decoder._normalize_blocks(observations['blocks'], n)
    p_dead, p_team = decoder._prediction_components(prediction, n)
    if dead is None:
        dead = decoder.dead_mask(p_dead, blocks, threshold=0.45, min_s=1.0)
    else:
        dead = np.asarray(dead)
        if dead.shape != (n,) or dead.dtype != np.bool_:
            raise ValueError('Dead override must be boolean (N,)')
    home, away = decoder._team_evidence(p_team, observations['X'], observations, SPEC)
    if transition_weights is None:
        costs = np.empty(0, dtype=np.float64)
        weights = None
    else:
        weights = np.asarray(transition_weights, dtype=np.float64)
        if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Transition weights must be positive finite (N,)')
        costs = np.empty(n, dtype=np.float64)
    rate = float(stats['rate'])
    if not np.isfinite(rate) or rate < 0:
        raise ValueError('Training rate must be finite and nonnegative')
    target = rate * sum(end - start for start, end in blocks) / (5 * 60.0)

    def make_labels(lam):
        if weights is not None:
            np.multiply(weights, lam, out=costs)
        codes = decoder._decode_live_team(home, away, blocks, dead, costs, lam)
        codes = decoder._merge_short_runs(codes, blocks, 5)
        if postprocess is not None:
            codes = np.asarray(postprocess(codes, observations, prediction))
            if codes.shape != (n,) or not np.isin(codes, (0, 1, 2)).all() or not np.array_equal(codes == 2, dead):
                raise ValueError('Postprocess must preserve shape, labels, and dead intervals')
        return codes.astype(np.int8, copy=False)

    lam = decoder.calibrate_lambda(make_labels, blocks, target)
    return make_labels(lam)


def load_bundle(name, regime, tag):
    needed = [previous.prediction_path(family, tag, name, regime) for family in ('short_tcn', 'boundary_tcn')]
    old_path = previous.prediction_path('old_pp', 'holdout', name, regime) if tag == 'holdout' else c.base.CACHE / 'predictions/pp_tcn' / tag / f'{name}_{regime}.npz'
    prior_root = c.CACHE if regime == 'stress' else c.base.CACHE
    needed += [old_path, prior_root / 'prior' / f'{name}_{regime}.npz']
    missing = [path for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required frozen probabilities missing: ' + ', '.join(str(path) for path in missing))
    memo = {}
    return {key: previous.probability(family, name, regime, tag, memo)
            for key, family in (('short', 'short_gru25'), ('old', 'old_gru25'), ('boundary', 'boundary_gru25'))}


def modules():
    return [c.import_path('wave6_' + path.stem, path) for path in sorted(HERE.glob('m0*.py'))]
