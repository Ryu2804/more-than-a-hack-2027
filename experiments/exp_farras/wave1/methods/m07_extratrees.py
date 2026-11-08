import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


CONFIGS = [
    {"name": "leaf5", "min_samples_leaf": 5},
    {"name": "leaf20", "min_samples_leaf": 20},
    {"name": "leaf60", "min_samples_leaf": 60},
]


def fit_predict(train, test, config):
    x_train = np.asarray(train["X"])
    y_train = np.asarray(train["y"])
    known_rows = np.flatnonzero(y_train >= 0)
    if known_rows.size == 0:
        raise ValueError("train has no labeled rows")

    if known_rows.size > 40000:
        known_rows = np.random.RandomState(2027).choice(
            known_rows, size=40000, replace=False
        )
        known_rows.sort()
        fit_x = x_train[known_rows]
        fit_y = y_train[known_rows]
    elif known_rows.size == y_train.size:
        fit_x = x_train
        fit_y = y_train
    else:
        fit_x = x_train[known_rows]
        fit_y = y_train[known_rows]

    model = ExtraTreesClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight="balanced",
        n_jobs=1,
        random_state=2027,
    )
    model.fit(fit_x, fit_y)

    x_test = np.asarray(test["X"])
    if x_test.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    probabilities = model.predict_proba(x_test)
    classes = model.classes_
    predictions = classes[np.argmax(probabilities, axis=1)].astype(np.int8, copy=False)

    for start, end in test["blocks"]:
        start = int(start)
        end = int(end)
        block_probabilities = probabilities[start:end]
        block_length = end - start
        if block_length <= 0:
            continue

        prefix = np.empty((block_length + 1, probabilities.shape[1]), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(block_probabilities, axis=0, out=prefix[1:])

        centers = np.arange(block_length)
        left = np.maximum(0, centers - 2)
        right = np.minimum(block_length, centers + 3)
        averages = (prefix[right] - prefix[left]) / (right - left)[:, None]
        predictions[start:end] = classes[np.argmax(averages, axis=1)].astype(
            np.int8, copy=False
        )

    return predictions
