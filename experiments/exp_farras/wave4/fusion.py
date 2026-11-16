import numpy as np


_CONFIG_WEIGHTS = (
    ("gru25", 0.25),
    ("gru50", 0.5),
    ("gru75", 0.75),
    ("tree25", 0.25),
)
CONFIGS = [{"name": name, "weight_other": weight} for name, weight in _CONFIG_WEIGHTS]
_WEIGHT_BY_NAME = dict(_CONFIG_WEIGHTS)
_FLOOR = 1e-8


def _numeric_array(value, label):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a real numeric array") from error
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{label} must be a real numeric array")
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a real numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _probability(prediction, label):
    try:
        value = prediction["probability"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"{label} must contain probability") from error
    probability = _numeric_array(value, f"{label} probability")
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError(f"{label} probability must have shape (N, 3)")
    if np.any(probability < 0):
        raise ValueError(f"{label} probability mass must be nonnegative")
    if np.any(np.max(probability, axis=1) <= 0):
        raise ValueError(f"{label} probability rows must have positive mass")
    return probability


def _boundary(prediction, label, n_samples):
    try:
        value = prediction["boundary"]
    except KeyError:
        return np.full(n_samples, 0.5, dtype=np.float64)
    except (TypeError, IndexError) as error:
        raise ValueError(f"{label} boundary must have shape (N,)") from error

    boundary = _numeric_array(value, f"{label} boundary")
    if boundary.shape != (n_samples,):
        raise ValueError(f"{label} boundary must have shape (N,)")
    if np.any((boundary < 0) | (boundary > 1)):
        raise ValueError(f"{label} boundary values must be in [0, 1]")
    return boundary


def combine(primary, other, config):
    if not isinstance(config, dict) or not isinstance(config.get("name"), str):
        raise ValueError("config must be a named configuration dictionary")
    name = config["name"]
    if name not in _WEIGHT_BY_NAME:
        raise ValueError(f"unknown fusion config: {name}")
    weight_other = _WEIGHT_BY_NAME[name]

    primary_probability = _probability(primary, "primary")
    other_probability = _probability(other, "other")
    if primary_probability.shape != other_probability.shape:
        raise ValueError("prediction probability shapes must match")

    n_samples = primary_probability.shape[0]
    primary_boundary = _boundary(primary, "primary", n_samples)
    other_boundary = _boundary(other, "other", n_samples)

    probability = (1.0 - weight_other) * primary_probability + weight_other * other_probability
    probability = np.maximum(probability, _FLOOR)
    probability /= np.sum(probability, axis=1, keepdims=True)

    p_dead = probability[:, 2]
    live_mass = probability[:, 0] + probability[:, 1]
    p_team = np.divide(
        probability[:, 0],
        live_mass,
        out=np.full(n_samples, 0.5, dtype=np.float64),
        where=live_mass > 0,
    )
    boundary = (1.0 - weight_other) * primary_boundary + weight_other * other_boundary

    return {
        "probability": probability,
        "p_dead": p_dead,
        "p_team": p_team,
        "boundary": boundary,
    }
