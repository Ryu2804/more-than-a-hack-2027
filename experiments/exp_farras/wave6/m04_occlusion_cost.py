import core


CONFIGS = [
    {'name': f'factor{factor:g}_window{window}', 'factor': factor, 'window': window}
    for factor in (1.25, 1.5, 2.0)
    for window in (1, 11)
]


def weights(observations, config):
    visibility = core.mean_by_half(
        observations['X'][:, 3], observations['blocks'], config['window']
    )
    return 1.0 + (config['factor'] - 1.0) * (1.0 - visibility)


def predict(bundle, observations, stats, config):
    return core.decode(
        bundle['short'], observations, stats,
        transition_weights=weights(observations, config),
    )
