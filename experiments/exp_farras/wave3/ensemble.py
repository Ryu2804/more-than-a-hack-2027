import numpy as np


CONFIGS = [
    {"name": "tree25", "tree_weight": 0.25},
    {"name": "tree50", "tree_weight": 0.5},
    {"name": "tree75", "tree_weight": 0.75},
    {
        "name": "visibility",
        "visible_tree_weight": 0.75,
        "hidden_tree_weight": 0.25,
    },
]


def combine(neural_prediction, tree_prediction, test, config):
    if not isinstance(config, dict) or not isinstance(config.get("name"), str):
        raise ValueError("config must be a named configuration dictionary")

    name = config["name"]
    if name == "tree25":
        tree_weight = 0.25
    elif name == "tree50":
        tree_weight = 0.5
    elif name == "tree75":
        tree_weight = 0.75
    elif name == "visibility":
        tree_weight = None
    else:
        raise ValueError(f"unknown ensemble config: {name}")

    def as_numeric_array(value):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError("array values must be real numbers")
        return array.astype(np.float64, copy=False)

    try:
        test_x = as_numeric_array(test["X"])
        neural_probability = as_numeric_array(neural_prediction["probability"])
        tree_probability = as_numeric_array(tree_prediction["probability"])
        neural_boundary = as_numeric_array(neural_prediction["boundary"])
        tree_boundary = as_numeric_array(tree_prediction["boundary"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("ensemble inputs must contain numeric arrays") from error

    if test_x.ndim != 2 or test_x.shape[1] != 16:
        raise ValueError("test X must have shape (N, 16)")
    if neural_probability.ndim != 2 or neural_probability.shape[1] != 3:
        raise ValueError("neural probability must have shape (N, 3)")
    if tree_probability.ndim != 2 or tree_probability.shape[1] != 3:
        raise ValueError("tree probability must have shape (N, 3)")

    n_samples = test_x.shape[0]
    if neural_probability.shape[0] != n_samples or tree_probability.shape[0] != n_samples:
        raise ValueError("prediction lengths must match test X")
    if neural_boundary.shape != (n_samples,) or tree_boundary.shape != (n_samples,):
        raise ValueError("boundary arrays must have shape (N,)")

    arrays = (
        test_x,
        neural_probability,
        tree_probability,
        neural_boundary,
        tree_boundary,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("ensemble inputs must be finite")
    if np.any(neural_probability < 0) or np.any(tree_probability < 0):
        raise ValueError("probabilities must be nonnegative")
    if np.any(neural_boundary < 0) or np.any(neural_boundary > 1):
        raise ValueError("neural boundary values must be in [0, 1]")
    if np.any(tree_boundary < 0) or np.any(tree_boundary > 1):
        raise ValueError("tree boundary values must be in [0, 1]")

    neural_scale = np.max(neural_probability, axis=1, keepdims=True)
    tree_scale = np.max(tree_probability, axis=1, keepdims=True)
    if np.any(neural_scale <= 0) or np.any(tree_scale <= 0):
        raise ValueError("each probability row must have positive mass")
    neural_probability = neural_probability / neural_scale
    tree_probability = tree_probability / tree_scale
    neural_probability /= np.sum(neural_probability, axis=1, keepdims=True)
    tree_probability /= np.sum(tree_probability, axis=1, keepdims=True)

    if name == "visibility":
        tree_weight = np.where(test_x[:, 3] > 0, 0.75, 0.25)
        probability_weight = tree_weight[:, None]
    else:
        probability_weight = tree_weight

    probability = (
        (1 - probability_weight) * neural_probability
        + probability_weight * tree_probability
    )
    probability = np.maximum(probability, 1e-8)
    probability /= np.sum(probability, axis=1, keepdims=True)
    boundary = (1 - tree_weight) * neural_boundary + tree_weight * tree_boundary

    return {"probability": probability, "boundary": boundary}
