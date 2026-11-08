from support import fit_emitter, probabilities, temporal, viterbi


CONFIGS = [
    {"name": "context1", "context_s": 1},
    {"name": "context3", "context_s": 3},
    {"name": "context7", "context_s": 7},
]


def fit(train, config):
    contextual = temporal(train, config["context_s"])
    return fit_emitter(contextual, kind="hist")


def predict(model, test, config):
    contextual = temporal(test, config["context_s"])
    p = probabilities(model, contextual)
    return viterbi(p, test["blocks"], switch_cost=6)
