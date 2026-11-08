from support import duration_decode, fit_emitter, probabilities


CONFIGS = [
    {"name": "dur04", "min_seconds": 0.4},
    {"name": "dur10", "min_seconds": 1.0},
    {"name": "dur20", "min_seconds": 2.0},
]


def fit(train, config):
    return fit_emitter(train, kind="hist")


def predict(model, test, config):
    minimum = config["min_seconds"]
    p = probabilities(model, test)
    return duration_decode(
        p,
        test["blocks"],
        fps=test["fps"],
        seconds=(minimum, minimum, minimum),
        switch_cost=6,
    )
