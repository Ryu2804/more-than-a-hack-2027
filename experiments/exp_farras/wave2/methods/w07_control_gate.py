import numpy as np

from support import fit_emitter, probabilities, viterbi


CONFIGS = [
    {'name': 'radius15', 'radius': 1.5},
    {'name': 'radius25', 'radius': 2.5},
    {'name': 'radius40', 'radius': 4.0},
]


def fit(train, config):
    return fit_emitter(train, kind='logistic')


def predict(model, test, config):
    p = probabilities(model, test)
    X = test['X']
    gated = (X[:, 3] < 0.5) | (np.minimum(X[:, 0], X[:, 1]) > config['radius'])
    p[gated, 0] = (1.0 - p[gated, 2]) / 2.0
    p[gated, 1] = (1.0 - p[gated, 2]) / 2.0
    return viterbi(p, test['blocks'], switch_cost=6)
