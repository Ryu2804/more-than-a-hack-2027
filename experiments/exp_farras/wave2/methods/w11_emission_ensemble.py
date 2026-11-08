from support import fit_emitter, probabilities, viterbi


CONFIGS = [
    {'name': 'hist25', 'hist_weight': 0.25},
    {'name': 'hist50', 'hist_weight': 0.5},
    {'name': 'hist75', 'hist_weight': 0.75},
]


def fit(train, config):
    return {
        'logistic': fit_emitter(train, kind='logistic'),
        'hist': fit_emitter(train, kind='hist'),
    }


def predict(model, test, config):
    p_logistic = probabilities(model['logistic'], test)
    p_hist = probabilities(model['hist'], test)
    weight = config['hist_weight']
    p = (1 - weight) * p_logistic + weight * p_hist
    return viterbi(p, test['blocks'], switch_cost=6)
