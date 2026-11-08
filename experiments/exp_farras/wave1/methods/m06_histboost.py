import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


CONFIGS = [
    {
        "name": "leaves7",
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 7,
        "l2_regularization": 1,
        "early_stopping": False,
        "random_state": 2027,
    },
    {
        "name": "leaves15",
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "l2_regularization": 1,
        "early_stopping": False,
        "random_state": 2027,
    },
    {
        "name": "leaves31",
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "l2_regularization": 1,
        "early_stopping": False,
        "random_state": 2027,
    },
]


_MAX_TRAIN_ROWS = 40000
_SMOOTH_RADIUS = 2


def fit_predict(train, test, config):
    train_y = np.asarray(train["y"])
    test_x = np.asarray(test["X"])
    if test_x.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    known_indices = np.flatnonzero(train_y >= 0)
    if known_indices.size == 0:
        raise ValueError("HistGradientBoostingClassifier requires known training labels")

    if known_indices.size > _MAX_TRAIN_ROWS:
        rng = np.random.default_rng(config["random_state"])
        fit_indices = np.sort(
            rng.choice(known_indices, size=_MAX_TRAIN_ROWS, replace=False)
        )
    else:
        fit_indices = known_indices

    fit_y = train_y[fit_indices]
    classes = np.unique(fit_y)
    if classes.size == 1:
        return np.full(test_x.shape[0], classes[0], dtype=np.int8)

    model_parameters = {
        key: value for key, value in config.items() if key != "name"
    }
    model = HistGradientBoostingClassifier(**model_parameters)
    model.fit(
        np.asarray(train["X"])[fit_indices],
        fit_y,
        sample_weight=compute_sample_weight(class_weight="balanced", y=fit_y),
    )

    raw_probabilities = model.predict_proba(test_x)
    probabilities = np.zeros((test_x.shape[0], 3), dtype=np.float64)
    for column, label in enumerate(model.classes_):
        probabilities[:, int(label)] = raw_probabilities[:, column]

    predictions = np.empty(test_x.shape[0], dtype=np.int8)
    for start, end in test["blocks"]:
        block_probabilities = probabilities[start:end]
        block_length = end - start
        prefix = np.empty((block_length + 1, 3), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(block_probabilities, axis=0, out=prefix[1:])
        positions = np.arange(block_length)
        left = np.maximum(positions - _SMOOTH_RADIUS, 0)
        right = np.minimum(positions + _SMOOTH_RADIUS + 1, block_length)
        averages = (prefix[right] - prefix[left]) / (right - left)[:, None]
        predictions[start:end] = np.argmax(averages, axis=1).astype(np.int8)

    return predictions
