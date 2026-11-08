"""Fixed feature/model primitives for the second, external-only selection campaign."""
import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 2027


def sample_rows(y, limit=40000):
    known = np.flatnonzero(y >= 0)
    if not len(known):
        raise ValueError('No known training labels')
    if len(known) <= limit:
        return known
    return np.sort(np.random.default_rng(SEED).choice(known, limit, replace=False))


def fit_emitter(train, kind='logistic', columns=None):
    X, y = train['X'], train['y']
    rows = sample_rows(y)
    X = X[rows] if columns is None else X[rows][:, columns]
    y = y[rows]
    classes = np.unique(y)
    if len(classes) == 1:
        return {'constant': int(classes[0]), 'columns': columns, 'model': None}
    if kind == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500,
                              class_weight='balanced', random_state=SEED))
        model.fit(X, y)
    elif kind == 'hist':
        model = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=100, learning_rate=.08,
                      l2_regularization=1, early_stopping=False, random_state=SEED)
        model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    elif kind == 'extra':
        model = ExtraTreesClassifier(n_estimators=120, max_depth=16, min_samples_leaf=20,
                                      class_weight='balanced', n_jobs=1, random_state=SEED)
        model.fit(X, y)
    elif kind == 'gaussian':
        model = GaussianNB(var_smoothing=1e-6)
        model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    else:
        raise ValueError(f'Unknown emitter: {kind}')
    return {'constant': None, 'columns': columns, 'model': model}


def probabilities(fitted, test):
    X = test['X']
    columns = fitted['columns']
    if columns is not None:
        X = X[:, columns]
    p = np.full((len(X), 3), 1e-8, dtype=np.float64)
    if fitted['constant'] is not None:
        p[:, fitted['constant']] = 1
    elif len(X):
        model = fitted['model']
        p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return p / p.sum(axis=1, keepdims=True)


def temporal(data, seconds):
    X = data['X']
    columns = [2, 3, 6, 8, 9]
    size = max(1, int(round(seconds * data['fps'])))
    size += int(size % 2 == 0)
    extra = np.empty((len(X), 10), np.float32)
    for a, b in data['blocks']:
        block = X[a:b, columns].astype(np.float64)
        mean = uniform_filter1d(block, size=size, axis=0, mode='nearest')
        variance = uniform_filter1d(block * block, size=size, axis=0, mode='nearest') - mean * mean
        extra[a:b, :5] = mean
        extra[a:b, 5:] = np.sqrt(np.maximum(variance, 0))
    return dict(data, X=np.column_stack([X, extra]))


def symmetrize(train):
    swapped = train['X'].copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    swapped[:, 2] *= -1
    y = train['y'].copy()
    alive = (y == 0) | (y == 1)
    y[alive] = 1 - y[alive]
    n = len(y)
    blocks = list(train['blocks']) + [(a + n, b + n) for a, b in train['blocks']]
    return dict(train, X=np.concatenate([train['X'], swapped]), y=np.r_[train['y'], y], blocks=blocks)


def transition_cost(train, strength=1):
    counts = np.ones((3, 3), dtype=np.float64)
    y = train['y']
    for a, b in train['blocks']:
        left, right = y[a:b - 1], y[a + 1:b]
        valid = (left >= 0) & (right >= 0)
        np.add.at(counts, (left[valid], right[valid]), 1)
    p = counts / counts.sum(axis=1, keepdims=True)
    return -float(strength) * np.log(p)


@njit(cache=False)
def _decode(logp, costs):
    n = len(logp)
    out = np.empty(n, np.int8)
    if n == 0:
        return out
    back = np.zeros((n, 3), np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        next_score = np.empty(3)
        c = costs[0 if len(costs) == 1 else t]
        for j in range(3):
            best, origin = score[j] - c[j, j], j
            for i in range(3):
                candidate = score[i] - c[i, j]
                if candidate > best:
                    best, origin = candidate, i
            next_score[j] = best + logp[t, j]
            back[t, j] = origin
        score = next_score
    out[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def viterbi(p, blocks, switch_cost=6, costs=None):
    if costs is None:
        costs = np.full((3, 3), float(switch_cost))
        np.fill_diagonal(costs, 0)
    costs = np.asarray(costs, np.float64)
    if costs.ndim == 2:
        costs = costs[None]
    logp = np.log(np.maximum(np.asarray(p, np.float64), 1e-8))
    out = np.empty(len(p), np.int8)
    for a, b in blocks:
        local = costs if len(costs) == 1 else costs[a:b]
        out[a:b] = _decode(logp[a:b], local)
    return out


@njit(cache=False)
def _duration_decode(logp, dwell, penalty):
    n = len(logp)
    if n == 0:
        return np.empty(0, np.int8)
    offsets = np.array([0, dwell[0], dwell[0] + dwell[1]])
    states = int(dwell.sum())
    mature = offsets + dwell - 1
    score = np.full(states, -np.inf)
    back = np.zeros((n, states), np.int16)
    for c in range(3):
        score[mature[c]] = logp[0, c]
    for t in range(1, n):
        updated = np.full(states, -np.inf)
        for c in range(3):
            start, end = offsets[c], mature[c]
            updated[end] = score[end] + logp[t, c]
            back[t, end] = end
            if dwell[c] > 1:
                for s in range(start + 1, end + 1):
                    candidate = score[s - 1] + logp[t, c]
                    if candidate > updated[s]:
                        updated[s] = candidate
                        back[t, s] = s - 1
            for other in range(3):
                if other != c:
                    candidate = score[mature[other]] - penalty + logp[t, c]
                    if candidate > updated[start]:
                        updated[start] = candidate
                        back[t, start] = mature[other]
        score = updated
    state = int(np.argmax(score))
    out = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = 0 if state < offsets[1] else (1 if state < offsets[2] else 2)
        state = back[t, state]
    return out


def duration_decode(p, blocks, fps=5, seconds=(1, 1, 1), switch_cost=6):
    dwell = np.maximum(1, np.ceil(np.asarray(seconds) * fps).astype(np.int64))
    logp = np.log(np.maximum(np.asarray(p, np.float64), 1e-8))
    out = np.empty(len(p), np.int8)
    for a, b in blocks:
        out[a:b] = _duration_decode(logp[a:b], dwell, float(switch_cost))
    return out
