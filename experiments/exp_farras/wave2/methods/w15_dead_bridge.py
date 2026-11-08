import numpy as np

from support import fit_emitter, probabilities, viterbi


CONFIGS = [
    {'name': 'bridge04', 'max_dead_s': 0.4},
    {'name': 'bridge10', 'max_dead_s': 1.0},
    {'name': 'bridge20', 'max_dead_s': 2.0},
]


def fit(train, config):
    return fit_emitter(train, kind='logistic')


def predict(model, test, config):
    labels = viterbi(probabilities(model, test), test['blocks'], switch_cost=6)
    bridged = labels.copy()
    max_length = config['max_dead_s'] * test['fps']

    for start, end in test['blocks']:
        i = start
        while i < end:
            j = i + 1
            while j < end and labels[j] == labels[i]:
                j += 1
            if labels[i] == 2 and i > start and j < end and j - i <= max_length:
                before, after = labels[i - 1], labels[j]
                if before == after and before in (0, 1):
                    bridged[i:j] = before
            i = j

    return bridged.astype(np.int8, copy=False)
