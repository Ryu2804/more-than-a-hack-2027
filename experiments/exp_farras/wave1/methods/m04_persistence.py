import numpy as np


CONFIGS = [
    {"name": "fast", "half_life_s": 0.5, "threshold": 2},
    {"name": "medium", "half_life_s": 1.5, "threshold": 4},
    {"name": "slow", "half_life_s": 3.0, "threshold": 6},
]


def fit_predict(train, test, config):
    X = test["X"]
    fps = float(test.get("fps", 5))
    half_life_s = float(config["half_life_s"])
    threshold = float(config["threshold"])
    decay = np.exp(-np.log(2.0) / (half_life_s * fps))
    score_limit = 2.0 * threshold
    predictions = np.empty(len(X), dtype=np.int8)

    for start, end in test["blocks"]:
        accumulator = 0.0
        first = X[start]
        team = 0 if first[3] < 0.5 or first[0] <= first[1] else 1
        in_dead = False

        for i in range(start, end):
            row = X[i]
            accumulator *= decay
            visible = row[3] >= 0.5
            controlled = visible and min(row[0], row[1]) <= 2.5
            dead_evidence = (
                (visible and row[7] < -0.3)
                or (
                    not visible
                    and row[14] + row[15] >= 10.0
                    and row[9] < 0.25
                )
            )

            if dead_evidence:
                in_dead = True
            elif controlled:
                margin = float(row[2])
                accumulator = float(
                    np.clip(accumulator + margin, -score_limit, score_limit)
                )
                if team == 0 and accumulator <= -threshold:
                    team = 1
                elif team == 1 and accumulator >= threshold:
                    team = 0
                in_dead = False

            if in_dead:
                predictions[i] = 2
            else:
                predictions[i] = team

    return predictions
