import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from numba import njit
except ImportError:
    njit = None


CONFIGS = [
    {"name": "dur04", "min_seconds": 0.4},
    {"name": "dur10", "min_seconds": 1.0},
    {"name": "dur20", "min_seconds": 2.0},
]

_MAX_TRAIN_ROWS = 40000
_SWITCH_COST = 6.0


def _sample_known_rows(y):
    known = np.flatnonzero(y >= 0)
    if known.size <= _MAX_TRAIN_ROWS:
        return known

    known_y = y[known]
    labels, counts = np.unique(known_y, return_counts=True)
    capacity = counts - 1
    extra_budget = _MAX_TRAIN_ROWS - labels.size
    shares = extra_budget * capacity / capacity.sum()
    extras = np.floor(shares).astype(np.int64)
    remaining = extra_budget - int(extras.sum())
    order = np.argsort(-(shares - extras), kind="stable")
    for index in order:
        if remaining == 0:
            break
        if extras[index] < capacity[index]:
            extras[index] += 1
            remaining -= 1

    rng = np.random.default_rng(2027)
    selected = []
    for label, count, extra in zip(labels, counts, extras):
        take = int(extra + 1)
        candidates = known[known_y == label]
        if take == count:
            selected.append(candidates)
        else:
            selected.append(rng.choice(candidates, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def _viterbi_block_impl(log_emission, dwell, switch_cost):
    n_frames, n_classes = log_emission.shape
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)

    n_states = n_classes * dwell
    back = np.empty((n_frames, n_states), dtype=np.uint8)
    previous = np.full(n_states, -np.inf, dtype=np.float64)
    current = np.empty(n_states, dtype=np.float64)

    for class_id in range(n_classes):
        mature = class_id * dwell + dwell - 1
        previous[mature] = log_emission[0, class_id]

    for frame in range(1, n_frames):
        current[:] = -np.inf
        for class_id in range(n_classes):
            base = class_id * dwell
            emission = log_emission[frame, class_id]

            if dwell == 1:
                best = previous[class_id]
                source = class_id
                for other in range(n_classes):
                    if other != class_id:
                        candidate = previous[other] - switch_cost
                        if candidate > best:
                            best = candidate
                            source = other
                current[class_id] = best + emission
                back[frame, class_id] = source
                continue

            best = previous[base + dwell - 1]
            source = base + dwell - 1
            candidate = previous[base + dwell - 2]
            if candidate > best:
                best = candidate
                source = base + dwell - 2
            current[base + dwell - 1] = best + emission
            back[frame, base + dwell - 1] = source

            for age in range(1, dwell - 1):
                state = base + age
                current[state] = previous[state - 1] + emission
                back[frame, state] = state - 1

            best = -np.inf
            source = 0
            for other in range(n_classes):
                if other != class_id:
                    candidate = previous[other * dwell + dwell - 1] - switch_cost
                    if candidate > best:
                        best = candidate
                        source = other * dwell + dwell - 1
            current[base] = best + emission
            back[frame, base] = source

        temporary = previous
        previous = current
        current = temporary

    state = int(np.argmax(previous))
    labels = np.empty(n_frames, dtype=np.int8)
    labels[n_frames - 1] = state // dwell
    for frame in range(n_frames - 1, 0, -1):
        state = int(back[frame, state])
        labels[frame - 1] = state // dwell
    return labels


_viterbi_block = njit(cache=False)(_viterbi_block_impl) if njit is not None else _viterbi_block_impl


def fit_predict(train, test, config):
    train_y = np.asarray(train["y"], dtype=np.int8)
    sample_rows = _sample_known_rows(train_y)
    if sample_rows.size == 0:
        raise ValueError("duration method requires at least one known training label")

    train_x = np.asarray(train["X"])[sample_rows]
    sample_y = train_y[sample_rows]
    test_x = np.asarray(test["X"])
    log_emission = np.full((test_x.shape[0], 3), -np.inf, dtype=np.float64)

    if np.unique(sample_y).size == 1:
        log_emission[:, int(sample_y[0])] = 0.0
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=400,
                random_state=2027,
                solver="lbfgs",
            ),
        )
        with threadpool_limits(limits=1):
            model.fit(train_x, sample_y)
            class_log_probability = model.predict_log_proba(test_x)
        log_emission[:, model.named_steps["logisticregression"].classes_] = class_log_probability

    predictions = np.argmax(log_emission, axis=1).astype(np.int8)
    fps = float(test.get("fps", 5.0))
    dwell = max(1, min(10, int(np.ceil(float(config["min_seconds"]) * fps - 1e-12))))
    for start, end in test["blocks"]:
        start = int(start)
        end = int(end)
        if end > start:
            predictions[start:end] = _viterbi_block(log_emission[start:end], dwell, _SWITCH_COST)
    return predictions
