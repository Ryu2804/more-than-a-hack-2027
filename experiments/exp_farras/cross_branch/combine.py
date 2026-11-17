import math
from collections.abc import Mapping

import numpy as np

import bridge as b


FAMILIES = (
    'short', 'setseq', 'pp', 'contact', 'memory', 'combined',
    'short_setseq25', 'short_pp25', 'short_contact25', 'short_memory25',
    'short_combined25', 'triple_contact', 'triple_combined', 'quality_combined',
)
DECODERS = ('dead45', 'hysteresis')
_SOURCES = FAMILIES[:6]


def _probability(bundle, name):
    return b.result(bundle[name]['probability'])['probability']


def visibility_mean(observations, n):
    """Centered eleven-sample visibility mean, clipped at half boundaries."""
    decoder = b.fair.decoder
    X = decoder._as_numeric(observations.get('X'), 'observations X')
    if X.ndim != 2 or X.shape != (n, 16):
        raise ValueError('observations X must have shape (N, 16)')
    visibility = decoder._vector(X[:, 3], 'visibility', n, probability=True)
    blocks = decoder._normalize_blocks(observations.get('blocks'), n)
    mean = np.empty(n, dtype=np.float64)
    for start, end in blocks:
        prefix = np.r_[0.0, np.cumsum(visibility[start:end], dtype=np.float64)]
        index = np.arange(end - start)
        left = np.maximum(0, index - 5)
        right = np.minimum(end - start, index + 6)
        mean[start:end] = (prefix[right] - prefix[left]) / (right - left)
    return np.clip(mean, 0.0, 1.0)


def fuse(name, bundle, observations):
    if name not in FAMILIES:
        raise ValueError(f'Unknown fusion family: {name}')
    if name in _SOURCES:
        return b.result(bundle[name]['probability'])

    short = _probability(bundle, 'short')
    if name.startswith('short_'):
        weights = ((name[len('short_'):-2], 0.25),)
        short_weight = 0.75
    elif name.startswith('triple_'):
        weights = (('setseq', 0.25), (name[len('triple_'):], 0.25))
        short_weight = 0.50
    else:
        combined_weight = 0.10 + 0.30 * visibility_mean(observations, len(short))
        weights = (('combined', combined_weight[:, None]),)
        short_weight = 1.0 - combined_weight[:, None]

    joint = short_weight * short
    for source, weight in weights:
        other = _probability(bundle, source)
        if other.shape != short.shape:
            raise ValueError('Fusion source probability shapes must match')
        joint += weight * other
    return b.result(joint)


def hysteresis_mask(p_dead, blocks, low=0.35, high=0.50):
    """Keep half-local components above low, length >=5, with a strict high seed."""
    decoder = b.fair.decoder
    values = decoder._as_numeric(p_dead, 'p_dead')
    if values.ndim != 1:
        raise ValueError('p_dead must have shape (N,)')
    values = decoder._vector(values, 'p_dead', len(values), probability=True)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError('Hysteresis thresholds must satisfy 0 <= low <= high <= 1')
    ranges = decoder._normalize_blocks(blocks, len(values))
    mask = np.zeros(len(values), dtype=bool)
    for start, end in ranges:
        local = values[start:end]
        edges = np.flatnonzero(np.diff(np.r_[False, local > low, False]))
        for left, right in edges.reshape(-1, 2):
            if right - left >= 5 and np.any(local[left:right] > high):
                mask[start + left:start + right] = True
    return mask


def decode(prediction, observations, stats, mode):
    if mode not in DECODERS:
        raise ValueError(f'Unknown decoder: {mode}')
    decoder = b.fair.decoder
    if mode == 'dead45':
        return decoder.decode(prediction, observations, b.fair.SPEC, stats)['codes']
    if not isinstance(observations, Mapping):
        raise ValueError('observations must be an observation dictionary')
    if not isinstance(stats, Mapping):
        raise ValueError('stats must be a dictionary')
    X = decoder._as_numeric(observations.get('X'), 'observations X')
    if X.ndim != 2 or X.shape[1] != 16:
        raise ValueError('observations X must have shape (N, 16)')
    blocks = decoder._normalize_blocks(observations.get('blocks'), len(X))
    p_dead, p_team = decoder._prediction_components(prediction, len(X))
    try:
        rate = float(stats['rate'])
        dead_share = float(stats['dead_share'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('stats must contain numeric rate and dead_share') from error
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError('training rate must be finite and nonnegative')
    if not math.isfinite(dead_share) or not 0.0 <= dead_share <= 1.0:
        raise ValueError('training dead_share must be in [0, 1]')

    dead = hysteresis_mask(p_dead, blocks)
    home, away = decoder._team_evidence(p_team, X, observations, b.fair.SPEC)
    target = rate * sum(end - start for start, end in blocks) / (decoder.FPS * 60.0)
    transition_costs = np.empty(0, dtype=np.float64)

    def make_labels(penalty):
        labels = decoder._decode_live_team(home, away, blocks, dead, transition_costs, penalty)
        return decoder._merge_short_runs(labels, blocks, 5)

    penalty = decoder.calibrate_lambda(make_labels, blocks, target)
    return make_labels(penalty).astype(np.int8, copy=False)
