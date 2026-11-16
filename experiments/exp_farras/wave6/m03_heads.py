import numpy as np

import core


CONFIGS = [
    {'name': f'{other}{int(weight * 100)}', 'other': other, 'weight_other': weight}
    for other in ('old', 'boundary')
    for weight in (0.25, 0.50, 0.75)
]


def fuse(bundle, config):
    short = bundle['short']
    other = bundle[config['other']]
    weight = config['weight_other']
    dead = ((1.0 - weight) * np.asarray(short['p_dead'], dtype=np.float64)
            + weight * np.asarray(other['p_dead'], dtype=np.float64))
    return core.from_heads(dead, np.asarray(short['p_team'], dtype=np.float64))


def predict(bundle, observations, stats, config):
    return core.decode(fuse(bundle, config), observations, stats)
