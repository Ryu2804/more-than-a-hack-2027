import numpy as np
from scipy.ndimage import convolve1d
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MAX_TRAIN_ROWS = 40000
RANDOM_STATE = 2027

CONFIGS = [
    {"name": "raw", "smooth_s": 0},
    {"name": "smooth1", "smooth_s": 1},
    {"name": "smooth3", "smooth_s": 3},
]


def _sample_known_rows(y):
    known = np.flatnonzero(y >= 0)
    if known.size <= MAX_TRAIN_ROWS:
        return known

    labels, counts = np.unique(y[known], return_counts=True)
    expected = counts * (MAX_TRAIN_ROWS / known.size)
    quotas = np.floor(expected).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)

    while quotas.sum() > MAX_TRAIN_ROWS:
        eligible = np.flatnonzero(quotas > 1)
        index = eligible[np.argmax(quotas[eligible] - expected[eligible])]
        quotas[index] -= 1

    remaining = MAX_TRAIN_ROWS - int(quotas.sum())
    while remaining:
        eligible = np.flatnonzero(quotas < counts)
        index = eligible[np.argmax(expected[eligible] - quotas[eligible])]
        quotas[index] += 1
        remaining -= 1

    rng = np.random.default_rng(RANDOM_STATE)
    selected = [
        rng.choice(np.flatnonzero(y == label), size=quota, replace=False)
        for label, quota in zip(labels, quotas)
    ]
    return np.sort(np.concatenate(selected))


def _window_frames(seconds, fps):
    frames = max(1, int(np.floor(seconds * fps + 0.5)))
    if frames > 1 and frames % 2 == 0:
        frames -= 1
    return frames


def _smooth_block(raw, window):
    if window <= 1 or raw.size <= 1:
        return raw.copy()

    one_hot = (np.arange(3)[:, None] == raw[None, :]).astype(np.uint8)
    counts = convolve1d(
        one_hot,
        weights=np.ones(window, dtype=np.uint8),
        axis=1,
        mode="nearest",
    )
    majority = counts.max(axis=0)
    result = counts.argmax(axis=0).astype(np.int8)
    positions = np.arange(raw.size)
    keep_raw = counts[raw, positions] == majority
    result[keep_raw] = raw[keep_raw]
    return result


def fit_predict(train, test, config):
    train_x = np.asarray(train["X"])
    train_y = np.asarray(train["y"])
    test_x = np.asarray(test["X"])
    if test_x.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    sample_indices = _sample_known_rows(train_y)
    if sample_indices.size == 0:
        raise ValueError("no known training labels")

    sample_y = train_y[sample_indices]
    classes = np.unique(sample_y)
    if classes.size == 1:
        raw = np.full(test_x.shape[0], classes[0], dtype=np.int8)
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1,
                        class_weight="balanced",
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        model.fit(train_x[sample_indices], sample_y)
        classifier = model.named_steps["classifier"]
        probabilities = model.predict_proba(test_x)
        raw = classifier.classes_[np.argmax(probabilities, axis=1)].astype(np.int8)

    window = _window_frames(config["smooth_s"], test["fps"])
    if window <= 1:
        return raw

    result = raw.copy()
    for start, end in test["blocks"]:
        result[start:end] = _smooth_block(raw[start:end], window)
    return result
