import numpy as np

import core


CONFIGS = [
    {'name': f'low{int(low * 100)}_high{int(high * 100)}', 'low': low, 'high': high}
    for low in (0.35, 0.40)
    for high in (0.50, 0.55, 0.60)
]


def hysteresis_mask(p_dead, blocks, low, high):
    """Retain half-local runs above low with five samples and a strict high seed."""
    values = np.asarray(p_dead)
    mask = np.zeros(len(values), dtype=bool)
    for start, end in blocks:
        local = values[start:end]
        edges = np.flatnonzero(np.diff(np.r_[False, local > low, False]))
        for left, right in edges.reshape(-1, 2):
            if right - left >= 5 and np.any(local[left:right] > high):
                mask[start + left:start + right] = True
    return mask


def predict(bundle, observations, stats, config):
    short = bundle['short']
    dead = hysteresis_mask(short['p_dead'], observations['blocks'], config['low'], config['high'])
    return core.decode(short, observations, stats, dead=dead)
