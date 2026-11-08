from support import fit_emitter, probabilities, transition_cost, viterbi


CONFIGS = [
    {'name': 'strength05', 'strength': 0.5},
    {'name': 'strength10', 'strength': 1},
    {'name': 'strength20', 'strength': 2},
]


def fit(train, config):
    return {
        'emitter': fit_emitter(train, kind='gaussian'),
        'costs': transition_cost(train, strength=config['strength']),
    }


def predict(model, test, config):
    p = probabilities(model['emitter'], test)
    return viterbi(p, test['blocks'], costs=model['costs'])
