"""H2: explicit-duration (semi-Markov) decoder with label-estimated duration priors.

A segment of state s (H, A or D) covering d steps at 5 fps scores

    sum_t log p_T(s | t)  +  w * (log P_s(d) + entropy(P_s))  -  lambda(previous state -> s)

The entropy term centres the duration score (zero mean under P_s): the raw log-pmf is about
-3 to -5 nats for every segment, so un-centred it acts as an extra switch penalty and lowers
the possession count, the opposite of what H2 is for.

with lambda_team for H<->A and lambda_dead for any change touching D. A segment may be
followed by one of the same state only if it has the maximum length D_MAX (continuation, so
spells longer than D_MAX stay possible; pieces merge in the output). Without that limit a long
spell could be cut into short same-state pieces to dodge the duration prior.
P_team (H and A pooled) and P_dead are discretised log-normals whose
parameters are the mean / SD of log run lengths in the *training* matches' labels of the
model whose predictions are decoded. With w = 0 this is exactly the constant-penalty
Viterbi of exp_1 (gamma = 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from numba import njit
from scipy.stats import norm

import exp1_decode as D
from exp1_common import expand, score

D_MAX = 1100  # steps (220 s); longest IDSSE dead spell is 207 s
DURATION_WEIGHTS = (0.25, 0.5, 1.0)
HSMM_LAMBDAS = (0.0, 0.5, 1.0, 2.0)
MIN_SIGMA = 0.1


@dataclass(frozen=True)
class DurationPrior:
    team_mu: float
    team_sigma: float
    dead_mu: float
    dead_sigma: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HsmmConfig:
    temperature: float
    weight: float
    lambda_team: float
    lambda_dead: float

    def as_dict(self) -> dict:
        return asdict(self)


def run_lengths(state: np.ndarray, half: np.ndarray) -> tuple[list[int], list[int]]:
    """Lengths of maximal team (H or A, same team) and dead runs per period; 3 breaks runs."""
    team, dead = [], []
    for period in np.unique(half):
        values = state[half == period]
        if not len(values):
            continue
        starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
        stops = np.r_[starts[1:], len(values)]
        for start, stop in zip(starts, stops):
            code = values[start]
            if code in (0, 1):
                team.append(int(stop - start))
            elif code == 2:
                dead.append(int(stop - start))
    return team, dead


def estimate_prior(labelled: list[tuple[np.ndarray, np.ndarray]]) -> DurationPrior:
    """Log-normal parameters of team and dead run lengths from (state, half) training pairs."""
    team, dead = [], []
    for state, half in labelled:
        t, d = run_lengths(state, half)
        team += t
        dead += d
    if not team or not dead:
        raise ValueError("Duration prior needs team and dead runs")
    log_team, log_dead = np.log(team), np.log(dead)
    return DurationPrior(float(log_team.mean()), max(MIN_SIGMA, float(log_team.std())),
                         float(log_dead.mean()), max(MIN_SIGMA, float(log_dead.std())))


def _log_pmf(mu: float, sigma: float, d_max: int) -> np.ndarray:
    d = np.arange(1, d_max + 1, dtype=np.float64)
    mass = norm.cdf((np.log(d + 0.5) - mu) / sigma) - norm.cdf((np.log(d - 0.5) - mu) / sigma)
    mass = np.clip(mass, 1e-300, None)
    return np.log(mass / mass.sum())


def _centred(log_pmf: np.ndarray) -> np.ndarray:
    """log P(d) + entropy(P): zero mean under P, so the term shapes durations without adding
    a constant per-segment cost (that cost is lambda's job)."""
    return log_pmf - float(np.sum(np.exp(log_pmf) * log_pmf))


def log_duration_table(prior: DurationPrior, weight: float, d_max: int = D_MAX) -> np.ndarray:
    """(3, d_max + 1) weighted, entropy-centred log P_s(d); column 0 is unused."""
    table = np.zeros((3, d_max + 1))
    team = _centred(_log_pmf(prior.team_mu, prior.team_sigma, d_max))
    table[0, 1:], table[1, 1:] = weight * team, weight * team
    table[2, 1:] = weight * _centred(_log_pmf(prior.dead_mu, prior.dead_sigma, d_max))
    table[:, 0] = -np.inf
    return table


@njit(cache=True)
def _hsmm(logp, logdur, lambda_team, lambda_dead):
    n = logp.shape[0]
    d_max = logdur.shape[1] - 1
    cum = np.zeros((n + 1, 3))
    for t in range(n):
        for s in range(3):
            cum[t + 1, s] = cum[t, s] + logp[t, s]
    best = np.full((n + 1, 3), -np.inf)  # best path over [0, t) whose last segment has state s
    full = np.full((n + 1, 3), -np.inf)  # same, but the last segment has length exactly d_max
    back_d = np.ones((n + 1, 3), dtype=np.int32)
    entry = np.full((n + 1, 3), -np.inf)  # best path over [0, t) that may start state s at t
    entry_from = np.zeros((n + 1, 3), dtype=np.int8)  # == s means continuation of a d_max piece
    for s in range(3):
        entry[0, s] = 0.0
    for t in range(1, n + 1):
        for s in range(3):
            value, length = -np.inf, 1
            for d in range(1, min(d_max, t) + 1):
                v = entry[t - d, s] + cum[t, s] - cum[t - d, s] + logdur[s, d]
                if v > value:
                    value, length = v, d
                if d == d_max:
                    full[t, s] = v
            best[t, s] = value
            back_d[t, s] = length
        if t < n:
            for s in range(3):
                value, source = full[t, s], s
                for p in range(3):
                    if p == s:
                        continue
                    penalty = lambda_dead if (p == 2 or s == 2) else lambda_team
                    v = best[t, p] - penalty
                    if v > value:
                        value, source = v, p
                entry[t, s] = value
                entry_from[t, s] = source
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax(best[n]))
    t = n
    d = back_d[t, state]
    while t > 0:
        path[t - d:t] = state
        t -= d
        if t == 0:
            break
        previous = int(entry_from[t, state])
        if previous == state:
            d = d_max
        else:
            state = previous
            d = back_d[t, state]
    return path


def decode(probability: np.ndarray, half: np.ndarray, prior: DurationPrior,
           config: HsmmConfig) -> np.ndarray:
    """5 fps codes, decoded independently in each period."""
    if probability.shape != (len(half), 3):
        raise ValueError("Probabilities must be (steps, 3) aligned with periods")
    logp = np.log(np.clip(D.calibrate(probability.astype(np.float64), config.temperature), 1e-6, 1.0))
    table = log_duration_table(prior, config.weight)
    output = np.empty(len(half), dtype=np.int8)
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        output[index] = _hsmm(logp[index], table, config.lambda_team, config.lambda_dead)
    return output


def score_item(item: dict, config: HsmmConfig) -> dict:
    codes = decode(item["probability"], item["half"], item["prior"], config)
    return score(item["truth"], expand(codes, item["indices"], item["full_half"]))


def select_config(validation: list[dict]) -> HsmmConfig:
    """Grid search on validation items only; each item carries the prior of its own model."""
    temperature = D.fit_temperature([v["probability"] for v in validation],
                                    [v["state"] for v in validation])
    best, best_value = None, -np.inf
    for weight, lambda_team, lambda_dead in product(DURATION_WEIGHTS, HSMM_LAMBDAS, HSMM_LAMBDAS):
        config = HsmmConfig(temperature, weight, lambda_team, lambda_dead)
        value = D.objective([score_item(v, config) for v in validation])
        if value > best_value + 1e-12:
            best, best_value = config, value
    if best is None:
        raise RuntimeError("HSMM grid produced no finite objective")
    return best
