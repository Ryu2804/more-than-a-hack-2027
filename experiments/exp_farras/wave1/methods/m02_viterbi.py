from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONFIGS = [
    {"name": "low", "switch_cost": 2},
    {"name": "medium", "switch_cost": 6},
    {"name": "high", "switch_cost": 12},
]


def _decode_block(
    log_emissions: np.ndarray,
    predictions: np.ndarray,
    offset: int,
    switch_cost: float,
) -> None:
    length = log_emissions.shape[0]
    if length == 0:
        return

    back = np.empty((length, 3), dtype=np.int8)
    score0, score1, score2 = log_emissions[0]
    next_scores = np.empty(3, dtype=np.float64)

    for t in range(1, length):
        if score0 >= score1:
            if score0 >= score2:
                best_state, best_score = 0, score0
                if score1 >= score2:
                    second_state, second_score = 1, score1
                else:
                    second_state, second_score = 2, score2
            else:
                best_state, best_score = 2, score2
                second_state, second_score = 0, score0
        elif score1 >= score2:
            best_state, best_score = 1, score1
            if score0 >= score2:
                second_state, second_score = 0, score0
            else:
                second_state, second_score = 2, score2
        else:
            best_state, best_score = 2, score2
            second_state, second_score = 1, score1

        other_score = second_score if best_state == 0 else best_score
        if score0 >= other_score - switch_cost:
            next_scores[0] = score0 + log_emissions[t, 0]
            back[t, 0] = 0
        else:
            previous = second_state if best_state == 0 else best_state
            next_scores[0] = other_score - switch_cost + log_emissions[t, 0]
            back[t, 0] = previous

        other_score = second_score if best_state == 1 else best_score
        if score1 >= other_score - switch_cost:
            next_scores[1] = score1 + log_emissions[t, 1]
            back[t, 1] = 1
        else:
            previous = second_state if best_state == 1 else best_state
            next_scores[1] = other_score - switch_cost + log_emissions[t, 1]
            back[t, 1] = previous

        other_score = second_score if best_state == 2 else best_score
        if score2 >= other_score - switch_cost:
            next_scores[2] = score2 + log_emissions[t, 2]
            back[t, 2] = 2
        else:
            previous = second_state if best_state == 2 else best_state
            next_scores[2] = other_score - switch_cost + log_emissions[t, 2]
            back[t, 2] = previous

        score0, score1, score2 = next_scores

    if score0 >= score1:
        state = 0 if score0 >= score2 else 2
    else:
        state = 1 if score1 >= score2 else 2

    predictions[offset + length - 1] = state
    for t in range(length - 1, 0, -1):
        state = int(back[t, state])
        predictions[offset + t - 1] = state


def fit_predict(train, test, config) -> np.ndarray:
    train_x = np.asarray(train["X"])
    train_y = np.asarray(train["y"])
    known = np.flatnonzero(train_y >= 0)
    if known.size == 0:
        raise ValueError("Viterbi logistic model requires known training labels")

    if known.size > 40000:
        rng = np.random.default_rng(2027)
        known = np.sort(rng.choice(known, size=40000, replace=False))

    fit_y = train_y[known]
    if np.any((fit_y < 0) | (fit_y > 2)):
        raise ValueError("Training labels must use states 0, 1, and 2")
    fit_y = fit_y.astype(np.int8, copy=False)
    if np.unique(fit_y).size < 2:
        raise ValueError("Viterbi logistic model requires at least two training classes")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1,
            class_weight="balanced",
            max_iter=400,
            n_jobs=1,
            random_state=2027,
        ),
    )
    model.fit(train_x[known], fit_y)

    test_x = np.asarray(test["X"])
    predictions = np.empty(test_x.shape[0], dtype=np.int8)
    expected_start = 0
    switch_cost = float(config["switch_cost"])
    classifier = model.named_steps["logisticregression"]
    for start, end in test["blocks"]:
        start, end = int(start), int(end)
        if start != expected_start or end < start or end > test_x.shape[0]:
            raise ValueError("Test blocks must partition test rows in order")
        if end > start:
            raw_probabilities = model.predict_proba(test_x[start:end])
            log_emissions = np.full((end - start, 3), 1e-6, dtype=np.float64)
            for probability_index, state in enumerate(classifier.classes_):
                log_emissions[:, int(state)] = raw_probabilities[:, probability_index]
            np.maximum(log_emissions, 1e-6, out=log_emissions)
            np.log(log_emissions, out=log_emissions)
            _decode_block(log_emissions, predictions, start, switch_cost)
        expected_start = end
    if expected_start != test_x.shape[0]:
        raise ValueError("Test blocks must cover every test row")
    return predictions
