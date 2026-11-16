"""Dataset-agnostic decoding from per-frame model outputs to H/A/D/X labels.

Used by the submission pipeline (run.py) and by the offline validation harness (experiments/016), so what is validated on other
matches is exactly the code that produces the submission.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent,):
    sys.path.insert(0, str(p))

import common  # noqa: E402

FPS, L, W = 25, 105.3, 68.0


def nearest_distances(P, B, T):
    """Nearest-player distance to the ball (metres) per team, shape (N, 2) = [home, away]; NaN when unknown.
    P (N,K,2+) normalised x,y; B (N,2+) normalised; T (K,) 0 home / 1 away."""
    n = len(B)
    out = np.full((n, 2), np.nan, np.float32)
    for a in range(0, n, 20000):
        b = min(n, a + 20000)
        d = np.hypot(P[a:b, :, 0] * L - (B[a:b, 0] * L)[:, None], P[a:b, :, 1] * W - (B[a:b, 1] * W)[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        for t in (0, 1):
            dt = np.where((T == t)[None], d, np.inf).min(1)
            out[a:b, t] = np.where(np.isfinite(dt), dt, np.nan)
    return out


def ball_evidence(dist, gate_r, tau=3.0):
    """Log-evidence for H and A from the nearest-player margin, counted only where the nearest player is within gate_r metres."""
    dH, dA = dist[:, 0], dist[:, 1]
    obs = np.isfinite(dH) & np.isfinite(dA)
    near = np.where(obs, np.minimum(dH, dA), np.inf)
    e = np.zeros(len(dH)); e[obs] = np.clip((dA[obs] - dH[obs]) / tau, -10, 10)
    lp_h, lp_a = -np.logaddexp(0, -e), -np.logaddexp(0, e)
    use = obs & (near <= gate_r)
    return np.where(use, lp_h, 0.0), np.where(use, lp_a, 0.0)


def dead_mask(p_dead, blocks, thr=0.5, min_s=2.0, shift=0.0):
    """Frames in dead-ball stretches: prior-shifted p(dead) > thr for at least min_s seconds, inside the graded blocks."""
    pc = np.clip(p_dead, 1e-4, 1 - 1e-4)
    p = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + shift)))
    m = np.zeros(len(p_dead), bool)
    for _, a, b in blocks:
        s, e = common.runs(np.nan_to_num(p[a:b + 1]) > thr)
        for si, ei in zip(s, e):
            if ei - si >= min_s * FPS:
                m[a + si:a + ei] = True
    return m


def evidence(lp_ball_h, lp_ball_a, p_team, kappa):
    pc = np.clip(p_team, 1e-3, 1 - 1e-3)
    return lp_ball_h + kappa * np.log(pc), lp_ball_a + kappa * np.log(1 - pc)


def decode(n_frames, blocks, h_ev, a_ev, lam, dead, forced_dead=()):
    """Two-state Viterbi per block, then D over dead stretches; X outside blocks. forced_dead: (first, last) frame ranges set to D."""
    lab = np.full(n_frames, "X", dtype="<U1")
    for _, a, b in blocks:
        path = common.viterbi_2state(h_ev[a:b + 1], a_ev[a:b + 1], lam)
        lab[a:b + 1] = np.where(path == 0, "H", "A")
    for a, b in forced_dead:
        lab[a:b] = "D"
    graded = np.zeros(n_frames, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    lab[dead & graded] = "D"
    return lab


def n_possessions(lab, blocks):
    return sum(len(common.runs(lab[a:b + 1] == t)[0]) for _, a, b in blocks for t in ("H", "A"))


def calibrate_lambda(make_labels, blocks, target, lambdas=(0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)):
    """lambda at which the number of possessions equals target (log-linear interpolation; clipped to the grid)."""
    ys = np.array([n_possessions(make_labels(l), blocks) for l in lambdas], float)
    x = np.log(lambdas)
    if target >= ys[0]:
        return float(lambdas[0])
    if target <= ys[-1]:
        return float(lambdas[-1])
    return float(np.exp(np.interp(-target, -ys, x)))


def viterbi_2state_t(lp_h, lp_a, cost):
    """2-state Viterbi with a per-frame switch cost array (nats)."""
    n = len(lp_h); em = np.stack([lp_h, lp_a], 1).astype(np.float64); back = np.zeros((n, 2), np.int8)
    s0, s1 = em[0]
    for t in range(1, n):
        c = cost[t]; a0, a1 = s0, s1
        if a0 >= a1 - c: b0, p0 = a0, 0
        else: b0, p0 = a1 - c, 1
        if a1 >= a0 - c: b1, p1 = a1, 1
        else: b1, p1 = a0 - c, 0
        back[t, 0], back[t, 1] = p0, p1; s0, s1 = b0 + em[t, 0], b1 + em[t, 1]
    path = np.zeros(n, np.int8); path[-1] = 0 if s0 >= s1 else 1
    for t in range(n - 1, 0, -1): path[t - 1] = back[t, path[t]]
    return path


def decode_t(n_frames, blocks, h_ev, a_ev, lam, dead, p_change, gamma):
    """decode() with switch cost lam * (1 - gamma * p_change_t): switching is cheap where the model expects a possession change."""
    lab = np.full(n_frames, "X", dtype="<U1")
    for _, a, b in blocks:
        cost = lam * (1 - gamma * np.nan_to_num(p_change[a:b + 1]))
        path = viterbi_2state_t(h_ev[a:b + 1], a_ev[a:b + 1], cost)
        lab[a:b + 1] = np.where(path == 0, "H", "A")
    graded = np.zeros(n_frames, bool)
    for _, a, b in blocks: graded[a:b + 1] = True
    lab[dead & graded] = "D"
    return lab
