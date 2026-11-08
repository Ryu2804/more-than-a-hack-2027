import numpy as np

from support import fit_emitter, probabilities, viterbi


CONFIGS = [
    {'name': 'factor1', 'factor': 1},
    {'name': 'factor2', 'factor': 2},
    {'name': 'factor4', 'factor': 4},
]


def fit(train, config):
    return fit_emitter(train, kind='logistic')


def predict(model, test, config):
    p = probabilities(model, test)
    costs = np.full((len(test['X']), 3, 3), 6.0, dtype=np.float64)
    costs[:, np.arange(3), np.arange(3)] = 0
    occluded = test['X'][:, 3] == 0
    costs[occluded, 0, 1] *= config['factor']
    costs[occluded, 1, 0] *= config['factor']
    return viterbi(p, test['blocks'], costs=costs)
