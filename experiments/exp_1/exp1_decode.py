"""Boundary-aware Viterbi decoder and train-only configuration selection.

Path score = sum_t log p_T(state_t) + transition terms, with
    stay   (i -> i): gamma * log(1 - b_t)
    switch (i -> j): -lambda_ij + gamma * log(b_t)
where b_t is the predicted boundary probability and lambda_ij is lambda_team for
H<->A and lambda_dead for any change touching D. gamma = 0 recovers the constant
penalty decoder used by exp_0 and rasya's DP, which is the ablation arm.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

from exp1_common import expand, score

LAMBDAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
GAMMAS = (0.0, 0.5, 1.0, 2.0)
COUNT_TOLERANCE = 0.30
COUNT_WEIGHT = 0.15
EPS = 1e-4


@dataclass(frozen=True)
class DecoderConfig:
    temperature: float
    lambda_team: float
    lambda_dead: float
    gamma: float

    def as_dict(self) -> dict:
        return asdict(self)


@njit(cache=True)
def _viterbi(logp, log_b, log_stay, lambda_team, lambda_dead, gamma):
    n = logp.shape[0]
    back = np.zeros((n, 3), dtype=np.int8)
    cost = logp[0].copy()
    for t in range(1, n):
        new = np.empty(3)
        for j in range(3):
            best, arg = -1e300, 0
            for i in range(3):
                if i == j:
                    value = cost[i] + gamma * log_stay[t]
                else:
                    penalty = lambda_dead if (i == 2 or j == 2) else lambda_team
                    value = cost[i] - penalty + gamma * log_b[t]
                if value > best:
                    best, arg = value, i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        cost = new
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax(cost))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def calibrate(probability: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probability, 1e-6, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    values = np.exp(logp)
    return values / values.sum(axis=1, keepdims=True)


def decode(probability: np.ndarray, boundary: np.ndarray | None, half: np.ndarray,
           config: DecoderConfig) -> np.ndarray:
    """5 fps codes, decoded independently in each period."""
    if probability.shape != (len(half), 3):
        raise ValueError("Probabilities must be (steps, 3) aligned with periods")
    b = np.full(len(half), 0.5) if boundary is None else np.clip(boundary, EPS, 1 - EPS)
    b = b.astype(np.float64)
    calibrated = calibrate(probability.astype(np.float64), config.temperature)
    logp = np.log(np.clip(calibrated, 1e-6, 1.0))
    output = np.empty(len(half), dtype=np.int8)
    for period in np.unique(half):
        index = np.flatnonzero(half == period)
        output[index] = _viterbi(
            logp[index], np.log(b[index]), np.log1p(-b[index]),
            config.lambda_team, config.lambda_dead, config.gamma,
        )
    return output


def fit_temperature(probabilities: list[np.ndarray], states: list[np.ndarray]) -> float:
    p = np.concatenate(probabilities)
    y = np.concatenate(states).astype(int)
    keep = y < 3
    p, y = p[keep], y[keep]

    def nll(temperature: float) -> float:
        picked = calibrate(p, temperature)[np.arange(len(y)), y]
        return float(-np.mean(np.log(np.clip(picked, 1e-7, 1.0))))

    return float(minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded").x)


def objective(results: list[dict]) -> float:
    """exp_0's frozen objective: spellF1 minus a penalty for count error beyond 30%."""
    mean_f1 = float(np.mean([r["spell_f1"] for r in results]))
    excess = float(np.mean([max(0.0, abs(r["count_error_ratio"]) - COUNT_TOLERANCE) for r in results]))
    return mean_f1 - COUNT_WEIGHT * excess


def select_config(validation: list[dict], gammas: tuple[float, ...]) -> tuple[DecoderConfig, list[dict]]:
    """Grid search on validation matches only.

    Each validation item holds: probability, boundary (or None), half, indices,
    full_half, truth (25 fps labels) and state (5 fps labels for calibration).
    """
    temperature = fit_temperature(
        [v["probability"] for v in validation], [v["state"] for v in validation]
    )
    search, best, best_value = [], None, -np.inf
    for gamma, lambda_team, lambda_dead in product(gammas, LAMBDAS, LAMBDAS):
        config = DecoderConfig(temperature, lambda_team, lambda_dead, gamma)
        results = [score_item(v, config) for v in validation]
        value = objective(results)
        search.append({**config.as_dict(), "objective": value,
                       "spell_f1": float(np.mean([r["spell_f1"] for r in results]))})
        if value > best_value + 1e-12:
            best, best_value = config, value
    if best is None:
        raise RuntimeError("Decoder grid produced no finite objective")
    return best, search


def score_item(item: dict, config: DecoderConfig) -> dict:
    codes = decode(item["probability"], item.get("boundary"), item["half"], config)
    return score(item["truth"], expand(codes, item["indices"], item["full_half"]))
