import math

import numpy as np


CONFIGS = [
    {"name": "r15", "radius": 1.5, "confirm_s": 0.4, "gap_dead_s": 8},
    {"name": "r25", "radius": 2.5, "confirm_s": 0.8, "gap_dead_s": 12},
    {"name": "r35", "radius": 3.5, "confirm_s": 1.2, "gap_dead_s": 16},
]


def fit_predict(train, test, config):
    X = np.asarray(test["X"])
    predictions = np.zeros(X.shape[0], dtype=np.int8)
    radius = float(config["radius"])
    confirm_frames = max(1, math.ceil(float(config["confirm_s"]) * 5))
    gap_dead_s = float(config["gap_dead_s"])

    for start, end in test["blocks"]:
        block = X[start:end]
        if block.shape[0] == 0:
            continue

        home_distance = block[:, 0]
        away_distance = block[:, 1]
        visible = block[:, 3] > 0.5
        proposals = np.where(home_distance <= away_distance, 0, 1)
        controlled = visible & (np.minimum(home_distance, away_distance) <= radius)
        dead_evidence = (visible & (block[:, 7] < -0.3)) | (
            ~visible
            & ((block[:, 14] + block[:, 15]) >= gap_dead_s)
            & (block[:, 8] < 1.0)
        )

        state = int(proposals[0]) if visible[0] else 0
        pending_team = -1
        pending_frames = 0

        for offset in range(block.shape[0]):
            proposal = int(proposals[offset])
            if dead_evidence[offset]:
                state = 2
                pending_team = -1
                pending_frames = 0
            elif state == 2:
                if controlled[offset]:
                    if pending_team == proposal:
                        pending_frames += 1
                    else:
                        pending_team = proposal
                        pending_frames = 1
                    if pending_frames >= confirm_frames:
                        state = proposal
                        pending_team = -1
                        pending_frames = 0
                else:
                    pending_team = -1
                    pending_frames = 0
            elif controlled[offset] and proposal != state:
                if pending_team == proposal:
                    pending_frames += 1
                else:
                    pending_team = proposal
                    pending_frames = 1
                if pending_frames >= confirm_frames:
                    state = proposal
                    pending_team = -1
                    pending_frames = 0
            else:
                pending_team = -1
                pending_frames = 0

            predictions[start + offset] = state

    return predictions
