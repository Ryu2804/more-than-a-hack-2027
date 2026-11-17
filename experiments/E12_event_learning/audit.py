from collections.abc import Mapping
from numbers import Integral

import numpy as np

import bridge as b

FAMILIES = ('balanced', 'focal', 'boundary', 'ranking', 'ema', 'contact')
SEEDS = (0, 1, 2)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def training(heldout, family, names, seed_fits, steps, expected_parameters):
    if family not in FAMILIES:
        raise ValueError('Unknown event learning family')
    if not _integer(steps) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not _integer(expected_parameters) or expected_parameters < 1:
        raise ValueError('Expected parameter count must be a positive integer')
    expected_names = b.fair.split_names(heldout)
    try:
        actual_names = tuple(names)
        records = tuple(seed_fits)
    except TypeError as error:
        raise ValueError('Training names and seed fits must be sequences') from error
    if actual_names != expected_names:
        raise ValueError('Training fold must contain exactly six other matches in split order')
    if len(records) != len(SEEDS):
        raise ValueError('Training requires exactly three seed fits')
    for seed, record in zip(SEEDS, records):
        if not isinstance(record, Mapping):
            raise ValueError('Each seed fit must be a mapping')
        values = tuple(record.get(key) for key in ('seed', 'updates', 'parameters'))
        if not all(_integer(value) for value in values):
            raise ValueError('Seed, updates and parameters must be integers')
        if values != (seed, steps, expected_parameters):
            raise ValueError('Seed order, update budget or parameter count mismatch')


def observations(fitted, observed):
    if not isinstance(observed, Mapping):
        raise ValueError('Prediction observations must be a mapping')
    if 'y' in observed or 'y25' in observed:
        raise ValueError('Prediction observations must not include GT')
    name = observed.get('_name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Prediction observations require a nonempty _name')
    if not isinstance(fitted, Mapping):
        raise ValueError('Fitted metadata must be a mapping')
    if isinstance(fitted.get('training_names'), (str, bytes)):
        raise ValueError('Training requires six unique match names')
    try:
        names = tuple(fitted.get('training_names', ()))
    except TypeError as error:
        raise ValueError('Training requires six unique match names') from error
    if (len(names) != 6 or
            any(not isinstance(value, str) or not value.strip() for value in names) or
            len(set(names)) != 6):
        raise ValueError('Training requires six unique match names')
    if name in names:
        raise ValueError('Prediction match was included in training')
    return observed


def probability(values, n):
    if not _integer(n) or n < 0:
        raise ValueError('Probability grid length must be a nonnegative integer')
    try:
        joint = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid joint probability array') from error
    if joint.shape != (n, 3) or joint.dtype.kind not in 'biuf':
        raise ValueError('Joint probabilities must have shape (N, 3) on the expected grid')
    if not np.isfinite(joint).all() or np.any(joint < 0):
        raise ValueError('Joint probabilities must be finite and nonnegative')
    mass = joint.sum(axis=1, dtype=np.float64)
    if np.any(mass <= 0) or np.any(np.abs(mass - 1.0) > 1e-6):
        raise ValueError('Joint probability rows must have positive mass normalized within 1e-6')
    return joint


def replay(actual, expected):
    try:
        actual = np.asarray(actual)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid replay probability array') from error
    actual = probability(actual, actual.shape[0] if actual.ndim else 0)
    expected = probability(expected, actual.shape[0])
    if not np.array_equal(actual, expected):
        raise ValueError('Saved-model inference differs from original probabilities')
