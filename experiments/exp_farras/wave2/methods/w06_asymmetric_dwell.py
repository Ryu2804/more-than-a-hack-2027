from support import duration_decode, fit_emitter, probabilities


CONFIGS = [
    {'name': 'dead04', 'dead_seconds': 0.4},
    {'name': 'dead10', 'dead_seconds': 1},
    {'name': 'dead20', 'dead_seconds': 2},
]


def fit(train, config):
    return fit_emitter(train, kind='logistic')


def predict(model, test, config):
    p = probabilities(model, test)
    return duration_decode(
        p,
        test['blocks'],
        fps=test['fps'],
        seconds=(1, 1, config['dead_seconds']),
        switch_cost=6,
    )
