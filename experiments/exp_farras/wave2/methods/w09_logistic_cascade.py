import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from support import SEED, sample_rows, viterbi


CONFIGS = [
    {'name': 'dead05', 'dead_strength': 0.5},
    {'name': 'dead10', 'dead_strength': 1.0},
    {'name': 'dead20', 'dead_strength': 2.0},
]


def _fit_binary(X, y):
    classes = np.unique(y)
    if len(classes) == 1:
        return int(classes[0]), None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1,
            max_iter=500,
            class_weight='balanced',
            random_state=SEED,
        ),
    )
    with threadpool_limits(limits=1):
        model.fit(X, y)
    return None, model


def fit(train, config):
    X = np.asarray(train['X'])
    y = np.asarray(train['y'])
    dead_rows = sample_rows(y, limit=40000)
    dead_target = (y[dead_rows] == 2).astype(np.int8)
    dead_constant, dead_model = _fit_binary(X[dead_rows], dead_target)

    alive_y = np.where((y == 0) | (y == 1), y, -1).astype(np.int8, copy=False)
    if np.any(alive_y >= 0):
        alive_rows = sample_rows(alive_y, limit=40000)
        alive_constant, alive_model = _fit_binary(X[alive_rows], y[alive_rows])
        all_dead = False
    else:
        alive_constant, alive_model = None, None
        all_dead = True

    return {
        'dead_constant': dead_constant,
        'dead_model': dead_model,
        'alive_constant': alive_constant,
        'alive_model': alive_model,
        'all_dead': all_dead,
    }


def _class_probability(model, X, class_value):
    class_index = np.flatnonzero(model.classes_ == class_value).item()
    with threadpool_limits(limits=1):
        return model.predict_proba(X)[:, class_index]


def predict(model, test, config):
    X = np.asarray(test['X'])
    n_rows = len(X)
    if model['dead_model'] is None:
        pd = np.full(n_rows, model['dead_constant'], dtype=np.float64)
    else:
        pd = _class_probability(model['dead_model'], X, 1)
    pd = expit(float(config['dead_strength']) * logit(pd))

    p = np.empty((n_rows, 3), dtype=np.float64)
    if model['all_dead']:
        p[:, 0] = 0.0
        p[:, 1] = 0.0
    else:
        if model['alive_model'] is None:
            ph = np.full(n_rows, 1.0 - model['alive_constant'], dtype=np.float64)
        else:
            ph = _class_probability(model['alive_model'], X, 0)
        p[:, 0] = (1.0 - pd) * ph
        p[:, 1] = (1.0 - pd) * (1.0 - ph)
    p[:, 2] = pd
    return viterbi(p, test['blocks'], switch_cost=6)
