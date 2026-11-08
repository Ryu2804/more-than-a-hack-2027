from support import fit_emitter, probabilities, viterbi


CONFIGS = [
    {'name': 'control', 'columns': [0, 1, 2, 3, 7, 14, 15]},
    {'name': 'shape', 'columns': [3, 8, 9, 10, 11, 12, 13, 14, 15]},
    {'name': 'no_position', 'columns': [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 15]},
]


def fit(train, config):
    return fit_emitter(train, kind='hist', columns=config['columns'])


def predict(model, test, config):
    p = probabilities(model, test)
    return viterbi(p, test['blocks'], switch_cost=6)
