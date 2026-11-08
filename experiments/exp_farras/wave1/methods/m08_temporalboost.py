import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier


_CONTEXT_COLUMNS = np.array([2, 3, 6, 8, 9], dtype=np.intp)
_MAX_TRAIN_ROWS = 40000

CONFIGS = [
    {"name": "context1", "context_s": 1},
    {"name": "context3", "context_s": 3},
    {"name": "context7", "context_s": 7},
]


def _make_features(data, context_s):
    x = np.asarray(data["X"], dtype=np.float32)
    window_size = max(1, int(round(context_s * data["fps"])))
    if window_size % 2 == 0:
        window_size += 1
    features = np.empty((x.shape[0], 26), dtype=np.float32)
    features[:, :16] = x
    features[:, 16:21] = x[:, _CONTEXT_COLUMNS]
    features[:, 21:26] = 0.0

    for start, end in data["blocks"]:
        if start >= end:
            continue
        block = x[start:end, _CONTEXT_COLUMNS]
        mean = uniform_filter1d(block, size=window_size, axis=0, mode="nearest")
        mean_square = uniform_filter1d(block * block, size=window_size, axis=0, mode="nearest")
        variance = np.maximum(mean_square - mean * mean, 0.0)
        features[start:end, 16:21] = mean
        features[start:end, 21:26] = np.sqrt(variance)

    return features


def _sample_known_rows(y):
    known = np.flatnonzero(y >= 0)
    if known.size <= _MAX_TRAIN_ROWS:
        return known

    labels, counts = np.unique(y[known], return_counts=True)
    exact = counts * (_MAX_TRAIN_ROWS / known.size)
    quotas = np.maximum(np.floor(exact).astype(np.int64), 1)

    while quotas.sum() > _MAX_TRAIN_ROWS:
        reducible = np.flatnonzero(quotas > 1)
        index = reducible[np.argmax(quotas[reducible] - exact[reducible])]
        quotas[index] -= 1

    remaining = _MAX_TRAIN_ROWS - int(quotas.sum())
    fractional = exact - np.floor(exact)
    while remaining:
        candidates = np.flatnonzero(quotas < counts)
        order = candidates[np.argsort(-fractional[candidates], kind="stable")]
        for index in order:
            quotas[index] += 1
            remaining -= 1
            if remaining == 0:
                break

    rng = np.random.default_rng(2027)
    sampled = []
    for label, quota in zip(labels, quotas):
        class_rows = known[y[known] == label]
        sampled.append(rng.choice(class_rows, size=int(quota), replace=False))
    return np.sort(np.concatenate(sampled))


def fit_predict(train, test, config):
    test_x = np.asarray(test["X"], dtype=np.float32)
    if test_x.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    context_s = config["context_s"]
    train_features = _make_features(train, context_s)
    test_features = _make_features(test, context_s)
    y = np.asarray(train["y"], dtype=np.int8)
    rows = _sample_known_rows(y)
    if rows.size == 0:
        raise ValueError("train contains no known labels")

    y_fit = y[rows]
    classes, inverse, class_counts = np.unique(y_fit, return_inverse=True, return_counts=True)
    if classes.size == 1:
        return np.full(test_x.shape[0], classes[0], dtype=np.int8)

    sample_weight = (y_fit.size / (classes.size * class_counts[inverse])).astype(np.float64)
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=15,
        max_iter=100,
        learning_rate=0.08,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=2027,
    )
    model.fit(train_features[rows], y_fit, sample_weight=sample_weight)
    return model.predict(test_features).astype(np.int8, copy=False)
