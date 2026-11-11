"""Viterbi linker over per-frame candidates (PREREG 3.5), the tau rule (3.6) and quality (3.7).

Pure functions on in-memory arrays; nothing is read from or written to disk here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

import exp6_common as C  # noqa: F401  (environment first)
from exp1_common import JD

TAU_GRID = tuple(round(0.1 * k, 1) for k in range(1, 10))
TAU_MIN_PRECISION = 0.95
TAU_FALLBACK = 0.9
INLIER_M = 2.0


@dataclass(frozen=True)
class LinkerConfig:
    max_speed_mps: float = 40.0  # 1.6 m per frame at 25 fps
    fps: int = 25
    none_score: float = 0.5  # emission of "none" = -log(0.5)
    switch_cost: float = 2.0  # candidate <-> none
    min_run: int = 5  # accepted frames sit in on-path candidate runs of >= 5 frames

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def max_step_m(self) -> float:
        return self.max_speed_mps / self.fps


def segments(frames: np.ndarray, half: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index ranges of consecutive video frames inside one half (``frames`` sorted)."""
    if not len(frames):
        return []
    breaks = np.flatnonzero((np.diff(frames) != 1) | (half[frames[1:]] != half[frames[:-1]])) + 1
    edges = np.r_[0, breaks, len(frames)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def viterbi(xy: np.ndarray, score: np.ndarray, cfg: LinkerConfig) -> np.ndarray:
    """Minimum-cost state per frame: candidate index 0..k-1, or k for none."""
    n, k = score.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = np.isfinite(score) & (score > 0) & np.isfinite(xy).all(axis=-1)
        emit = np.where(valid, -np.log(np.where(valid, score, 1.0)), np.inf)
    none = -math.log(cfg.none_score)
    cost = np.r_[emit[0], none]
    back = np.zeros((n, k + 1), dtype=np.int16)
    columns = np.arange(k)
    for t in range(1, n):
        with np.errstate(invalid="ignore"):
            allowed = np.linalg.norm(xy[t][None, :, :] - xy[t - 1][:, None, :], axis=-1) <= cfg.max_step_m
        from_candidate = np.where(allowed, cost[:k, None], np.inf)  # (previous, current)
        best = from_candidate.argmin(axis=0)
        value = from_candidate[best, columns]
        via_none = cost[k] + cfg.switch_cost
        take_none = via_none < value
        back[t, :k] = np.where(take_none, k, best)
        new = np.empty(k + 1)
        new[:k] = np.where(take_none, via_none, value) + emit[t]
        to_none = cost[:k] + cfg.switch_cost
        j = int(to_none.argmin())
        back[t, k] = k if cost[k] <= to_none[j] else j
        new[k] = min(cost[k], to_none[j]) + none
        cost = new
    state = np.empty(n, dtype=np.int16)
    state[-1] = int(cost.argmin())
    for t in range(n - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return state


def link(frames: np.ndarray, xy: np.ndarray, score: np.ndarray, half: np.ndarray,
         cfg: LinkerConfig) -> dict:
    """One pass per run of consecutive inferred frames within a half (non-inferred frames = none)."""
    n, k = score.shape
    state = np.full(n, k, dtype=np.int16)
    run_length = np.zeros(n, dtype=np.int64)
    for a, b in segments(frames, half):
        state[a:b] = viterbi(xy[a:b], score[a:b], cfg)
        for start, stop in JD.runs(state[a:b] < k):
            run_length[a + start:a + stop] = stop - start
    on = state < k
    position = np.full((n, 2), np.nan)
    position_score = np.full(n, np.nan)
    rows = np.flatnonzero(on)
    position[rows] = xy[rows, state[rows]]
    position_score[rows] = score[rows, state[rows]]
    return {"frame": frames, "state": state, "position": position, "score": position_score,
            "run_length": run_length, "k": k}


def accept(linked: dict, tau: float, cfg: LinkerConfig) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (linked["state"] < linked["k"]) & (linked["score"] >= tau) & (linked["run_length"] >= cfg.min_run)


def quality(accepted: np.ndarray, position: np.ndarray, truth: np.ndarray) -> dict:
    """precision@2 m, recall@2 m, per-axis RMS of inliers, f = 1 - precision, on frames with truth."""
    error = np.linalg.norm(position - truth, axis=1)
    with np.errstate(invalid="ignore"):
        inlier = accepted & (error <= INLIER_M)
    n_accepted, n_inlier = int(accepted.sum()), int(inlier.sum())
    precision = n_inlier / n_accepted if n_accepted else float("nan")
    residual = (position - truth)[inlier]
    return {"frames": int(len(accepted)), "accepted": n_accepted, "inliers": n_inlier,
            "precision": float(precision), "recall": float(n_inlier / max(1, len(accepted))),
            "sigma_m": float(np.sqrt(np.mean(residual ** 2))) if n_inlier else float("nan"),
            "fp_rate": float(1.0 - precision) if n_accepted else float("nan"),
            "median_error_inliers_m": float(np.median(error[inlier])) if n_inlier else float("nan")}


def select_tau(linked: dict, truth: np.ndarray, val: np.ndarray, cfg: LinkerConfig) -> tuple[float, list[dict]]:
    """Smallest tau in the grid with val precision@2 m >= 0.95; otherwise 0.9 (the gates decide)."""
    table = []
    for tau in TAU_GRID:
        q = quality(accept(linked, tau, cfg)[val], linked["position"][val], truth[val])
        table.append({"tau": tau, **q})
    passing = [row["tau"] for row in table if row["precision"] >= TAU_MIN_PRECISION]
    return (min(passing) if passing else TAU_FALLBACK), table
