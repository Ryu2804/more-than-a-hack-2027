from support import fit_emitter, probabilities, symmetrize, viterbi


CONFIGS = [
    {'name': 'cost2', 'switch_cost': 2},
    {'name': 'cost6', 'switch_cost': 6},
    {'name': 'cost12', 'switch_cost': 12},
]


def fit(train, config):
    return fit_emitter(symmetrize(train), kind='hist')


def predict(model, test, config):
    p = probabilities(model, test)
    return viterbi(p, test['blocks'], switch_cost=config['switch_cost'])
