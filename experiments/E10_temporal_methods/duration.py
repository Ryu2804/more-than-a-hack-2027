"""Supervised 5fps duration priors and exact, right-censored age-state Viterbi."""

import numpy as np
from numba import njit

import bridge as b


CONFIGS = [
    dict(name=f'strength{strength:g}_bias{bias:g}', strength=strength, switch_bias=bias)
    for strength in (0.25, 0.5, 1.0)
    for bias in (-0.5, 0.0)
]


def _cap(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('cap must be an integer')
    value = int(value)
    if not 1 <= value <= 21845:
        raise ValueError('cap must be between 1 and 21845 for uint16 backpointers')
    return value


def fit(datasets, cap=80):
    """Omit both boundary runs of each half's contiguous known interval."""
    cap = _cap(cap)
    durations = [[], [], []]
    frame_counts = np.zeros(3, dtype=np.int64)
    transition_counts = np.zeros((3, 3), dtype=np.int64)
    names = []
    interval_count = 0
    censored_count = 0
    for data in datasets:
        labels = np.asarray(data['y'])
        if labels.ndim != 1 or not np.isin(labels, (-1, 0, 1, 2)).all():
            raise ValueError('training y must be a vector with labels -1, 0, 1, 2')
        blocks = b.decoder._normalize_blocks(data['blocks'], len(labels))
        names.append(str(data['_name']))
        for start, end in blocks:
            half = labels[start:end]
            frame_counts += np.array([np.count_nonzero(half == state) for state in range(3)])
            source, target = half[:-1], half[1:]
            changed = (source >= 0) & (target >= 0) & (source != target)
            np.add.at(transition_counts, (source[changed].astype(np.int64),
                                         target[changed].astype(np.int64)), 1)
            known = half >= 0
            edges = np.flatnonzero(np.diff(np.r_[False, known, False]))
            for left, right in zip(edges[::2], edges[1::2]):
                interval_count += 1
                interval = half[left:right]
                boundaries = np.r_[0, np.flatnonzero(interval[1:] != interval[:-1]) + 1,
                                   len(interval)]
                run_count = len(boundaries) - 1
                censored_count += min(run_count, 2)
                for run in range(1, run_count - 1):
                    a, z = boundaries[run:run + 2]
                    durations[int(interval[a])].append(int(z - a))
    if not names:
        raise ValueError('at least one training dataset is required')

    hazard = np.empty((3, cap), dtype=np.float64)
    ages = np.arange(1, cap)
    for state in range(3):
        lengths = np.asarray(durations[state], dtype=np.int64)
        histogram = np.bincount(np.minimum(lengths, cap), minlength=cap + 1)
        risk = np.cumsum(histogram[::-1])[::-1]
        hazard[state, :-1] = (histogram[ages] + 1.0) / (risk[ages] + 2.0)
        tail_count = np.count_nonzero(lengths >= cap)
        tail_exposure = np.maximum(lengths - cap + 1, 0).sum()
        hazard[state, -1] = (tail_count + 1.0) / (tail_exposure + 2.0)

    transition = transition_counts.astype(np.float64) + 1.0
    np.fill_diagonal(transition, 0.0)
    transition /= transition.sum(axis=1, keepdims=True)
    # Diagonal log entries are finite sentinels; self-transitions use survival only.
    log_transition = np.zeros((3, 3), dtype=np.float64)
    off_diagonal = ~np.eye(3, dtype=bool)
    log_transition[off_diagonal] = np.log(transition[off_diagonal])
    initial = (frame_counts + 1.0) / (frame_counts.sum() + 3.0)
    complete_counts = np.array([len(values) for values in durations], dtype=np.int64)
    return dict(
        cap=cap, fps=5, hazard=hazard, log_hazard=np.log(hazard),
        log_survival=np.log1p(-hazard), transition=transition,
        log_transition=log_transition, initial=initial, log_initial=np.log(initial),
        training_names=tuple(names), frame_counts=frame_counts,
        transition_counts=transition_counts, complete_counts=complete_counts,
        complete_durations=tuple(tuple(values) for values in durations),
        known_interval_count=interval_count, censored_run_count=censored_count,
        missing_complete_states=tuple(int(state) for state in np.flatnonzero(complete_counts == 0)),
        hazard_prior='Beta(1,1)', tail='age-capped geometric',
    )


def _log_probability(probability):
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('probability must have shape (N, 3)')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('probability must be finite and nonnegative')
    total = values.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError('probability rows must have positive finite mass')
    return np.log(np.maximum(values / total, 1e-8))


@njit(cache=False)
def _viterbi_half(emission, log_initial, log_hazard, log_survival,
                  log_transition, strength, switch_bias):
    n = len(emission)
    cap = log_hazard.shape[1]
    size = 3 * cap
    path = np.empty(n, dtype=np.int8)
    if n == 0:
        return path
    back = np.empty((n, size), dtype=np.uint16)
    previous = np.full(size, -np.inf)
    current = np.empty(size)
    for state in range(3):
        previous[state * cap] = log_initial[state] + emission[0, state]
    stay = strength * log_survival
    leave = strength * log_hazard
    transition = strength * log_transition - switch_bias
    for time in range(1, n):
        current[:] = -np.inf
        # Ascending sources and strict comparisons select the lowest predecessor on ties.
        for source in range(size):
            score = previous[source]
            if score == -np.inf:
                continue
            state = source // cap
            age = source % cap
            target = state * cap + min(age + 1, cap - 1)
            candidate = score + stay[state, age]
            if candidate > current[target]:
                current[target] = candidate
                back[time, target] = source
            candidate_exit = score + leave[state, age]
            for destination in range(3):
                if destination == state:
                    continue
                target = destination * cap
                candidate = candidate_exit + transition[state, destination]
                if candidate > current[target]:
                    current[target] = candidate
                    back[time, target] = source
        for target in range(size):
            current[target] += emission[time, target // cap]
        previous, current = current, previous
    best = 0
    for index in range(1, size):
        if previous[index] > previous[best]:
            best = index
    path[n - 1] = best // cap
    for time in range(n - 1, 0, -1):
        best = int(back[time, best])
        path[time - 1] = best // cap
    return path


def viterbi(probability, blocks, prior, strength, switch_bias):
    """Return exact joint state paths, resetting age and initial scores per half."""
    emission = _log_probability(probability)
    blocks = b.decoder._normalize_blocks(blocks, len(emission))
    cap = _cap(prior['cap'])
    strength, switch_bias = float(strength), float(switch_bias)
    if not np.isfinite(strength) or strength < 0 or not np.isfinite(switch_bias):
        raise ValueError('strength must be finite and nonnegative; switch_bias must be finite')
    arrays = []
    for key, shape in (('log_initial', (3,)), ('log_hazard', (3, cap)),
                       ('log_survival', (3, cap)), ('log_transition', (3, 3))):
        value = np.asarray(prior[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'{key} must be finite with shape {shape}')
        arrays.append(np.ascontiguousarray(value))
    codes = np.empty(len(emission), dtype=np.int8)
    for start, end in blocks:
        codes[start:end] = _viterbi_half(emission[start:end], *arrays, strength, switch_bias)
    return codes


def decode(prediction, observations, prior, config):
    """Use only joint emissions and half boundaries, never held-out labels or gates."""
    return viterbi(prediction['probability'], observations['blocks'], prior,
                   config['strength'], config['switch_bias'])
