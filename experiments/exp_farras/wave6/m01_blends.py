import core


CONFIGS = [
    {'name': f'{other}{percent}', 'other': other, 'weight': percent / 100}
    for other in ('old', 'boundary')
    for percent in (10, 25, 50)
]


def fuse(bundle, config):
    return core.mix(bundle['short'], bundle[config['other']], config['weight'])


def predict(bundle, observations, stats, config):
    return core.decode(fuse(bundle, config), observations, stats)
