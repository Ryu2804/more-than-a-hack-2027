import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


CONTROL_COLUMNS = (0, 1, 2, 3, 7, 14, 15)
MAX_TRAIN_ROWS = 40000
SEED = 2027
PROBABILITY_FLOOR = 1e-8

CONFIGS = [
    {"name": "control", "columns": list(CONTROL_COLUMNS), "symmetric": False},
    {"name": "control_symmetric", "columns": list(CONTROL_COLUMNS), "symmetric": True},
    {"name": "all_features", "columns": list(range(16)), "symmetric": False},
]
_CONFIGS_BY_NAME = {config["name"]: config for config in CONFIGS}


def _resolve_config(config):
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError("config must be a CONFIGS dictionary")
    name = config["name"]
    if name not in _CONFIGS_BY_NAME:
        raise ValueError(f"Unknown tree config: {name}")
    return _CONFIGS_BY_NAME[name]


def _validate_features(values):
    features = np.asarray(values)
    if features.ndim != 2 or features.shape[1] != 16:
        raise ValueError("X must have shape (N, 16)")
    return features


def _concat_training_data(datasets):
    feature_parts = []
    label_parts = []
    blocks = []
    offset = 0
    for dataset in datasets:
        features = _validate_features(dataset["X"])
        labels = np.asarray(dataset["y"])
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("y must have shape (N,)")
        if np.any((labels < -1) | (labels > 2)):
            raise ValueError("y labels must be -1, 0, 1, or 2")
        feature_parts.append(features)
        label_parts.append(labels)
        for start, end in dataset["blocks"]:
            start, end = int(start), int(end)
            if start < 0 or end < start or end > len(features):
                raise ValueError("block range is outside dataset")
            if start < end:
                blocks.append((offset + start, offset + end))
        offset += len(features)

    if feature_parts:
        features = np.concatenate(feature_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
    else:
        features = np.empty((0, 16), dtype=np.float32)
        labels = np.empty(0, dtype=np.int8)
    return features, labels, blocks


def _sample_indices(size, limit):
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    selected = np.random.default_rng(SEED).choice(size, size=limit, replace=False)
    return np.sort(selected)


def _mirror_full_features(features):
    mirrored = features.copy()
    mirrored[:, [0, 1]] = mirrored[:, [1, 0]]
    mirrored[:, 2] *= -1
    return mirrored


def _mirror_emission_features(features, columns):
    positions = {column: index for index, column in enumerate(columns)}
    mirrored = features.copy()
    mirrored[:, [positions[0], positions[1]]] = mirrored[:, [positions[1], positions[0]]]
    mirrored[:, positions[2]] *= -1
    return mirrored


def _fit_classifier(features, labels):
    classes = np.unique(labels)
    if len(classes) == 1:
        return {"status": "constant", "constant": int(classes[0]), "model": None}

    model = HistGradientBoostingClassifier(
        max_leaf_nodes=15,
        max_iter=100,
        learning_rate=0.08,
        l2_regularization=1,
        early_stopping=False,
        random_state=SEED,
    )
    model.fit(features, labels, sample_weight=compute_sample_weight("balanced", labels))
    return {"status": "model", "constant": None, "model": model}


def _boundary_training_data(features, labels, blocks):
    feature_parts = []
    target_parts = []
    for start, end in blocks:
        if end - start < 2:
            continue
        current_rows = np.arange(start + 1, end, dtype=np.int64)
        known = (labels[current_rows - 1] >= 0) & (labels[current_rows] >= 0)
        current_rows = current_rows[known]
        if not len(current_rows):
            continue
        current = features[current_rows]
        delta = current - features[current_rows - 1]
        feature_parts.append(np.concatenate((current, delta), axis=1))
        target_parts.append((labels[current_rows] != labels[current_rows - 1]).astype(np.int8))

    if not feature_parts:
        return np.empty((0, 32), dtype=features.dtype), np.empty(0, dtype=np.int8)
    return np.concatenate(feature_parts, axis=0), np.concatenate(target_parts, axis=0)


def _fit_boundary(features, labels, blocks, symmetric):
    pair_features, pair_targets = _boundary_training_data(features, labels, blocks)
    if not len(pair_targets):
        return {"status": "prior_no_eligible_pairs", "prior": 0.5, "constant": None, "model": None}

    paired_limit = MAX_TRAIN_ROWS // 2 if symmetric else MAX_TRAIN_ROWS
    selected = _sample_indices(len(pair_targets), paired_limit)
    pair_features = pair_features[selected]
    pair_targets = pair_targets[selected]
    if symmetric:
        current = _mirror_full_features(pair_features[:, :16])
        delta = _mirror_full_features(pair_features[:, 16:])
        mirrored = np.concatenate((current, delta), axis=1)
        pair_features = np.concatenate((pair_features, mirrored), axis=0)
        pair_targets = np.concatenate((pair_targets, pair_targets), axis=0)

    fitted = _fit_classifier(pair_features, pair_targets)
    fitted["prior"] = None
    return fitted


def fit(datasets, config):
    selected_config = _resolve_config(config)
    features, labels, blocks = _concat_training_data(datasets)
    known_rows = np.flatnonzero(labels >= 0)
    if not len(known_rows):
        raise ValueError("No known training labels")

    symmetric = selected_config["symmetric"]
    emission_limit = MAX_TRAIN_ROWS // 2 if symmetric else MAX_TRAIN_ROWS
    emission_rows = known_rows[_sample_indices(len(known_rows), emission_limit)]
    columns = selected_config["columns"]
    emission_features = features[emission_rows][:, columns]
    emission_labels = labels[emission_rows].copy()
    if symmetric:
        mirrored_features = _mirror_emission_features(emission_features, columns)
        mirrored_labels = emission_labels.copy()
        alive = (mirrored_labels == 0) | (mirrored_labels == 1)
        mirrored_labels[alive] = 1 - mirrored_labels[alive]
        emission_features = np.concatenate((emission_features, mirrored_features), axis=0)
        emission_labels = np.concatenate((emission_labels, mirrored_labels), axis=0)

    emission = _fit_classifier(emission_features, emission_labels)
    boundary = _fit_boundary(features, labels, blocks, symmetric)
    return {
        "name": selected_config["name"],
        "columns": tuple(columns),
        "emission": emission,
        "boundary": boundary,
        "boundary_status": boundary["status"],
    }


def _state_probabilities(fitted, features):
    probabilities = np.full((len(features), 3), PROBABILITY_FLOOR, dtype=np.float64)
    if not len(features):
        return probabilities
    if fitted["status"] == "constant":
        probabilities[:, fitted["constant"]] = 1.0
    else:
        raw = np.asarray(fitted["model"].predict_proba(features), dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        for index, state in enumerate(fitted["model"].classes_):
            probabilities[:, int(state)] = np.maximum(raw[:, index], PROBABILITY_FLOOR)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _boundary_probability(fitted, features):
    if fitted["status"] == "prior_no_eligible_pairs":
        return np.full(len(features), fitted["prior"], dtype=np.float64)
    if fitted["status"] == "constant":
        return np.full(len(features), fitted["constant"], dtype=np.float64)
    raw = np.asarray(fitted["model"].predict_proba(features), dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)
    positive = np.flatnonzero(fitted["model"].classes_ == 1)
    if not len(positive):
        return np.zeros(len(features), dtype=np.float64)
    return np.clip(raw[:, int(positive[0])], 0.0, 1.0)


def predict(fitted, test, config):
    selected_config = _resolve_config(config)
    if fitted["name"] != selected_config["name"]:
        raise ValueError("fitted model and config names differ")
    features = _validate_features(test["X"])
    emission_features = features[:, fitted["columns"]]
    probabilities = _state_probabilities(fitted["emission"], emission_features)

    boundary = np.full(len(features), 0.5, dtype=np.float64)
    if fitted["boundary_status"] != "prior_no_eligible_pairs":
        for start, end in test["blocks"]:
            start, end = int(start), int(end)
            if start < 0 or end < start or end > len(features):
                raise ValueError("block range is outside test data")
            if end - start < 2:
                continue
            current_rows = np.arange(start + 1, end, dtype=np.int64)
            current = features[current_rows]
            delta = current - features[current_rows - 1]
            pair_features = np.concatenate((current, delta), axis=1)
            boundary[current_rows] = _boundary_probability(fitted["boundary"], pair_features)

    boundary = np.nan_to_num(boundary, nan=0.5, posinf=0.5, neginf=0.5)
    return {"probability": probabilities, "boundary": np.clip(boundary, 0.0, 1.0)}
