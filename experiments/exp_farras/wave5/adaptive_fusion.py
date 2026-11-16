import numpy as np


FAMILIES = (
    "old_pp",
    "old_gru25",
    "boundary_tcn",
    "short_tcn",
    "masked_tcn",
    "boundary_gru25",
    "short_gru25",
    "masked_gru25",
    "quality_old",
    "new_ensemble",
    "new_ensemble_gru25",
    "quality_new",
)

_FLOOR = 1e-8
_GRU_WEIGHT = 0.25


def _mapping(value, label):
    if not hasattr(value, "__getitem__"):
        raise ValueError(f"{label} must be a prediction mapping")
    return value


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


def _field(prediction, keys, label):
    for key in keys:
        try:
            return prediction[key]
        except (KeyError, TypeError, IndexError):
            pass
    raise ValueError(f"{label} must contain {keys[0]}")


def _probability(prediction, label):
    prediction = _mapping(prediction, label)
    value = _field(prediction, ("probability", "hybrid_probability"), label)
    probability = _numeric_array(value, f"{label} probability")
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError(f"{label} probability must have shape (N, 3)")
    if np.any(probability < 0):
        raise ValueError(f"{label} probability mass must be nonnegative")
    if np.any(np.max(probability, axis=1) <= 0):
        raise ValueError(f"{label} probability rows must have positive mass")
    return probability


def _has_boundary(prediction, label):
    try:
        value = prediction["has_boundary"]
    except (KeyError, TypeError, IndexError):
        return False
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} has_boundary must be a boolean")
    return bool(value)


def _boundary(prediction, label, n_samples, has_boundary):
    try:
        value = _field(prediction, ("boundary", "hybrid_boundary"), label)
    except ValueError:
        if has_boundary:
            raise ValueError(f"{label} has_boundary requires boundary values")
        return np.full(n_samples, 0.5, dtype=np.float64)

    boundary = _numeric_array(value, f"{label} boundary")
    if boundary.shape != (n_samples,):
        raise ValueError(f"{label} boundary must have shape (N,)")
    if np.any((boundary < 0) | (boundary > 1)):
        raise ValueError(f"{label} boundary values must be in [0, 1]")
    return boundary


def _conditionals(probability):
    p_dead = probability[:, 2]
    live_mass = probability[:, 0] + probability[:, 1]
    p_team = np.divide(
        probability[:, 0],
        live_mass,
        out=np.full(len(probability), 0.5, dtype=np.float64),
        where=live_mass > 0,
    )
    return p_dead, p_team


def _result(probability, boundary, has_boundary):
    probability = np.maximum(probability, _FLOOR)
    probability /= np.sum(probability, axis=1, keepdims=True)
    p_dead, p_team = _conditionals(probability)
    return {
        "probability": probability,
        "p_dead": p_dead,
        "p_team": p_team,
        "boundary": boundary,
        "has_boundary": bool(has_boundary),
    }


def _source_result(prediction, label, *, force_no_boundary=False):
    probability = _probability(prediction, label)
    has_boundary = False if force_no_boundary else _has_boundary(prediction, label)
    boundary = _boundary(prediction, label, len(probability), has_boundary)
    return _result(probability.copy(), boundary, has_boundary)


def _joint_mix(primary, other, weight_other):
    return (1.0 - weight_other) * primary + weight_other * other


def _blocks(observations, n_samples):
    try:
        raw = observations["blocks"]
    except (KeyError, TypeError, IndexError):
        if n_samples == 0:
            return []
        return [(0, n_samples)]

    try:
        bounds = np.asarray(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("observations blocks must contain start/end pairs") from error
    if bounds.size == 0 and n_samples == 0:
        return []
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("observations blocks must contain start/end pairs")

    blocks = []
    cursor = 0
    for start_value, end_value in bounds:
        try:
            start, end = int(start_value), int(end_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("observations blocks must use integer bounds") from error
        if start != start_value or end != end_value or start != cursor or end <= start or end > n_samples:
            raise ValueError("observations blocks must partition samples in order")
        blocks.append((start, end))
        cursor = end
    if cursor != n_samples:
        raise ValueError("observations blocks must cover every sample")
    return blocks


def _quality_weights(primary, observations):
    observations = _mapping(observations, "observations")
    try:
        x_value = observations["X"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("observations must contain X") from error
    try:
        x = np.asarray(x_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("observations X must be a real numeric matrix") from error
    if x.dtype.kind not in "biuf" or x.ndim != 2 or x.shape[1] <= 3:
        raise ValueError("observations X must be a real numeric matrix with column 3")
    visibility = _numeric_array(x[:, 3], "observations X[:, 3]")
    if len(visibility) != len(primary):
        raise ValueError("observations and prediction lengths must match")
    if np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("observations X[:, 3] must be in [0, 1]")

    blocks = _blocks(observations, len(primary))
    centered_visibility = np.empty(len(visibility), dtype=np.float64)
    prefix = np.empty(len(visibility) + 1, dtype=np.float64)
    for start, end in blocks:
        prefix[start] = 0.0
        np.cumsum(visibility[start:end], dtype=np.float64, out=prefix[start + 1 : end + 1])
        prefix[start + 1 : end + 1] += prefix[start]
        for index in range(start, end):
            left = max(start, index - 5)
            right = min(end, index + 6)
            centered_visibility[index] = (prefix[right] - prefix[left]) / (right - left)

    live_mass = primary[:, 0] + primary[:, 1]
    p_team = np.divide(
        primary[:, 0],
        live_mass,
        out=np.full(len(primary), 0.5, dtype=np.float64),
        where=live_mass > 0,
    )
    entropy = np.zeros(len(primary), dtype=np.float64)
    positive = p_team > 0
    below_one = p_team < 1
    entropy[positive] -= p_team[positive] * np.log(p_team[positive])
    entropy[below_one] -= (1.0 - p_team[below_one]) * np.log1p(-p_team[below_one])
    entropy /= np.log(2.0)
    return 0.05 + 0.35 * centered_visibility * entropy


def _quality_mix(primary, gru, observations, has_boundary, boundary):
    weight = _quality_weights(primary, observations)
    probability = (1.0 - weight[:, None]) * primary + weight[:, None] * gru
    return _result(probability, boundary, has_boundary)


def combine(family, base_predictions, old_pp, gru, observations):
    if family not in FAMILIES:
        raise ValueError(f"unknown fusion family: {family}")

    if family == "old_pp":
        return _source_result(old_pp, "old_pp", force_no_boundary=True)

    if family == "old_gru25":
        primary = _probability(old_pp, "old_pp")
        other = _probability(gru, "gru")
        if primary.shape != other.shape:
            raise ValueError("prediction probability shapes must match")
        boundary = _boundary(old_pp, "old_pp", len(primary), False)
        return _result(
            _joint_mix(primary, other, _GRU_WEIGHT),
            boundary,
            False,
        )

    if family in ("boundary_tcn", "short_tcn", "masked_tcn"):
        return _source_result(base_predictions[family], family)

    if family in ("boundary_gru25", "short_gru25", "masked_gru25"):
        primary_name = family[:-5] + "tcn"
        prediction = base_predictions[primary_name]
        primary = _probability(prediction, primary_name)
        other = _probability(gru, "gru")
        if primary.shape != other.shape:
            raise ValueError("prediction probability shapes must match")
        has_boundary = _has_boundary(prediction, primary_name)
        boundary = _boundary(prediction, primary_name, len(primary), has_boundary)
        return _result(
            _joint_mix(primary, other, _GRU_WEIGHT),
            boundary,
            has_boundary,
        )

    if family == "quality_old":
        primary = _probability(old_pp, "old_pp")
        other = _probability(gru, "gru")
        if primary.shape != other.shape:
            raise ValueError("prediction probability shapes must match")
        boundary = _boundary(old_pp, "old_pp", len(primary), False)
        return _quality_mix(primary, other, observations, False, boundary)

    if family in ("new_ensemble", "new_ensemble_gru25", "quality_new"):
        predictions = [_probability(base_predictions[name], name) for name in (
            "boundary_tcn", "short_tcn", "masked_tcn"
        )]
        if predictions[1].shape != predictions[0].shape or predictions[2].shape != predictions[0].shape:
            raise ValueError("prediction probability shapes must match")
        primary_probability = (predictions[0] + predictions[1] + predictions[2]) / 3.0
        boundary_prediction = base_predictions["boundary_tcn"]
        has_boundary = _has_boundary(boundary_prediction, "boundary_tcn")
        boundary = _boundary(
            boundary_prediction, "boundary_tcn", len(primary_probability), has_boundary
        )

        if family == "new_ensemble":
            return _result(primary_probability, boundary, has_boundary)

        other = _probability(gru, "gru")
        if primary_probability.shape != other.shape:
            raise ValueError("prediction probability shapes must match")
        if family == "new_ensemble_gru25":
            return _result(
                _joint_mix(primary_probability, other, _GRU_WEIGHT),
                boundary,
                has_boundary,
            )
        return _quality_mix(
            primary_probability, other, observations, has_boundary, boundary
        )

    raise AssertionError("unhandled fusion family")
