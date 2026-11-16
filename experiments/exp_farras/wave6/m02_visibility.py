import core


CONFIGS = [
    {
        'name': f'visible{round(visible_weight * 100)}_missing{round(missing_weight * 100)}',
        'visible_weight': visible_weight,
        'missing_weight': missing_weight,
    }
    for visible_weight in (0.85, 1.0)
    for missing_weight in (0.25, 0.50, 0.65)
]


def fuse(bundle, observations, config):
    visibility = core.mean_by_half(observations['X'][:, 3], observations['blocks'], 11)
    short_weight = config['missing_weight'] + (
        config['visible_weight'] - config['missing_weight']
    ) * visibility
    return core.mix(bundle['old'], bundle['short'], short_weight)


def predict(bundle, observations, stats, config):
    return core.decode(fuse(bundle, observations, config), observations, stats)
