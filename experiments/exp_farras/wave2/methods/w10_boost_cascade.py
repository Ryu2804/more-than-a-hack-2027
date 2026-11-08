import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from support import SEED, sample_rows, viterbi


CONFIGS = [
    {'name': 'dead05', 'dead_strength': 0.5},
    {'name': 'dead10', 'dead_strength': 1.0},
    {'name': 'dead20', 'dead_strength': 2.0},
]


def _fit_binary(X, y):
    classes = np.unique(y)
    if len(classes) == 1:
        return {'constant': int(classes[0]), 'model': None}
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=15,
        max_iter=100,
        learning_rate=0.08,
        l2_regularization=1,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    return {'constant': None, 'model': model}


def _positive_probability(fitted, X):
    if fitted['constant'] is not None:
        return np.full(len(X), fitted['constant'], dtype=np.float64)
    if not len(X):
        return np.empty(0, dtype=np.float64)
    model = fitted['model']
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, positive_column]


def _scale_dead_probability(pd, strength):
    pd = np.asarray(pd, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(pd, eps, 1 - eps)
    logit = np.log(clipped) - np.log1p(-clipped)
    scaled = 1 / (1 + np.exp(-strength * logit))
    scaled[pd <= 0] = 0
    scaled[pd >= 1] = 1
    return scaled


def fit(train, config):
    X, y = train['X'], train['y']
    known_rows = sample_rows(y, limit=40000)
    dead_target = (y[known_rows] == 2).astype(np.int8)
    dead_model = _fit_binary(X[known_rows], dead_target)
    all_dead = not np.any((y == 0) | (y == 1))

    if all_dead:
        alive_model = None
    else:
        alive_rows = np.flatnonzero((y == 0) | (y == 1))
        alive_rows = alive_rows[sample_rows(y[alive_rows], limit=40000)]
        alive_h_target = (y[alive_rows] == 0).astype(np.int8)
        alive_model = _fit_binary(X[alive_rows], alive_h_target)

    return {'dead': dead_model, 'alive': alive_model, 'all_dead': all_dead}


def predict(model, test, config):
    X = test['X']
    if model['all_dead']:
        p = np.zeros((len(X), 3), dtype=np.float64)
        p[:, 2] = 1
    else:
        pd = _positive_probability(model['dead'], X)
        pd = _scale_dead_probability(pd, config['dead_strength'])
        ph = _positive_probability(model['alive'], X)
        alive = 1 - pd
        p = np.empty((len(X), 3), dtype=np.float64)
        p[:, 0] = alive * ph
        p[:, 1] = alive * (1 - ph)
        p[:, 2] = pd
    return viterbi(p, test['blocks'], switch_cost=6)
