import numpy as np
from support import fit_emitter, probabilities


CONFIGS = [
    {'name': 'conf55', 'threshold': 0.55},
    {'name': 'conf70', 'threshold': 0.70},
    {'name': 'conf85', 'threshold': 0.85},
]


def fit(train, config):
    return fit_emitter(train, kind='logistic')


def predict(model, test, config):
    p = probabilities(model, test)
    labels = np.empty(len(p), dtype=np.int8)
    threshold = config['threshold']

    for start, end in test['blocks']:
        if start >= end:
            continue
        current = int(np.argmax(p[start]))
        labels[start] = current
        pending = -1
        pending_count = 0

        for i in range(start + 1, end):
            candidate = int(np.argmax(p[i]))
            if candidate != current and p[i, candidate] >= threshold:
                if candidate == pending:
                    pending_count += 1
                else:
                    pending = candidate
                    pending_count = 1
                if pending_count >= 3:
                    current = candidate
                    pending = -1
                    pending_count = 0
            else:
                pending = -1
                pending_count = 0
            labels[i] = current

    return labels
