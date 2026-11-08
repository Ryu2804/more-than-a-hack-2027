#!/usr/bin/env python3
"""C09 prototype: learned team attribution from tracking features.

The C07/C08 line showed the dead cue is not the bottleneck; the team is. The first prototype failed
because the features did not encode the side of the nearest player. This one does: per-team minima
and counts, plus ball state and touch history.

Step 1 (this script, default): frame-level accuracy on held-out match, before any decoder. If the
classifier is near-constant or does not beat nearest-player accuracy, stop and do not pre-register.
Step 2 (--decode): 2-state Viterbi on log p with switch penalty, then D from the frozen cue and
merge, scored with spellF1.

Exploratory only until pre-registered.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
import harness as H  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402
from deployable import build_prediction  # noqa: E402

R_M, MINLEN, MERGE_K = 0.8, 30, 150
CLIP = 30.0


def features(npz, ctx):
    px, py, played = npz["player_x"], npz["player_y"], npz["player_mask"]
    bx, by = npz["ball_x"], npz["ball_y"]
    side = H.sides(npz)
    if side is None:
        return None
    if ctx["flipped"]:
        side = 1 - side
    n = len(bx)
    d = np.hypot(px - bx[None, :], py - by[None, :])
    finite = np.isfinite(bx) & np.isfinite(by)
    d = np.where(played & finite[None, :], d, np.inf)

    home = side == 0
    away = side == 1
    mh = np.where(home[:, None], d, np.inf).min(axis=0)
    ma = np.where(away[:, None], d, np.inf).min(axis=0)
    mh2 = np.partition(np.where(home[:, None], d, np.inf), 1, axis=0)[1]
    ma2 = np.partition(np.where(away[:, None], d, np.inf), 1, axis=0)[1]
    nh = (d < 5.0)[home].sum(axis=0)
    na = (d < 5.0)[away].sum(axis=0)
    for arr in (mh, ma, mh2, ma2):
        arr[~np.isfinite(arr)] = CLIP

    speed = np.nan_to_num(npz["ball_speed"].astype(np.float64) / 3.6, nan=0.0, posinf=0.0)
    held = ctx["held"]
    age = np.zeros(n)
    last = np.full(n, 0.5)
    current, since = 0.5, 0
    for t in range(n):
        if held[t] >= 0 and (t == 0 or held[t] != held[t - 1]):
            current, since = float(held[t]), 0
        else:
            since += 1
        last[t] = current
        age[t] = min(since, 250)

    gap = ma - mh
    rolling = np.zeros(n)
    csum = np.cumsum(gap)
    for t in range(n):
        lo = max(0, t - 50)
        rolling[t] = (csum[t] - (csum[lo - 1] if lo else 0.0)) / (t - lo + 1)

    X = np.column_stack([
        np.minimum(mh, CLIP), np.minimum(ma, CLIP),
        np.minimum(mh2, CLIP), np.minimum(ma2, CLIP),
        np.clip(gap, -CLIP, CLIP), nh, na,
        np.minimum(speed, 30.0), bx / 105.0, by / 68.0,
        last, age / 250.0, np.clip(rolling, -CLIP, CLIP),
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def train(X, y, l2=1e-2):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    Xb = np.column_stack([np.ones(len(Xs)), Xs])

    def objective(w):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        return loss, grad

    result = minimize(objective, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300})
    return result.x, mean, std


def probabilities(w, mean, std, X):
    Xs = (X - mean) / std
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    z = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def viterbi(p, penalty):
    """Two-state decode of p(home). penalty in log units per switch."""
    n = len(p)
    eps = 1e-6
    log_h = np.log(np.clip(p, eps, 1 - eps))
    log_a = np.log(np.clip(1 - p, eps, 1 - eps))
    dp = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=np.int8)
    dp[0] = [log_h[0], log_a[0]]
    for t in range(1, n):
        stay = dp[t - 1]
        switch = dp[t - 1][::-1] - penalty
        best_home = stay[0] >= switch[0]
        dp[t, 0] = log_h[t] + (stay[0] if best_home else switch[0])
        back[t, 0] = 0 if best_home else 1
        best_away = stay[1] >= switch[1]
        dp[t, 1] = log_a[t] + (stay[1] if best_away else switch[1])
        back[t, 1] = 1 if best_away else 0
    state = int(np.argmax(dp[-1]))
    path = np.zeros(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = int(back[t, state])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    data = {}
    for name in H.DEV:
        ctx = T.match_context(name)
        npz = H.load(name)
        ctx["held"] = T.hold(ctx["dmin"], ctx["team"], R_M, MINLEN)
        ctx["side"] = ctx["team"]  # already flipped by match_context
        ctx["cue"], _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
        ctx["X"] = features(npz, ctx)
        data[name] = ctx

    rows = {}
    for held_out in H.DEV:
        train_names = [m for m in H.DEV if m != held_out]
        Xtr, ytr = [], []
        for name in train_names:
            ctx = data[name]
            alive = (ctx["gt"] == 0) | (ctx["gt"] == 1)
            Xtr.append(ctx["X"][alive])
            ytr.append((ctx["gt"][alive] == 1).astype(np.float64))
        w, mean, std = train(np.vstack(Xtr), np.concatenate(ytr))

        ctx = data[held_out]
        alive = (ctx["gt"] == 0) | (ctx["gt"] == 1)
        p = probabilities(w, mean, std, ctx["X"])
        accuracy = float(((p[alive] > 0.5) == (ctx["gt"][alive] == 1)).mean())
        nearest = float((ctx["team"][alive] == ctx["gt"][alive]).mean())
        row = {"accuracy": accuracy, "nearest_accuracy": nearest,
               "p_mean": float(p[alive].mean()), "p_std": float(p[alive].std())}
        print(held_out, "acc", round(accuracy, 4), "nearest", round(nearest, 4),
              "p_std", round(row["p_std"], 4), flush=True)

        if args.decode and accuracy >= 0.80:
            p_decode = p.copy()
            p_decode[ctx["cue"]] = 0.5  # model trained on alive frames; dead frames carry no evidence
            best = None
            for penalty in (20.0, 50.0, 100.0, 200.0, 500.0):
                path = viterbi(1.0 - p_decode, penalty)  # p is P(away); viterbi takes P(home)
                dead = ctx["cue"]
                pred = build_prediction(ctx, dead, path.astype(np.int8), MERGE_K)
                scores = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
                f1 = float(scores["mean_f1"])
                print("   penalty", penalty, "f1", round(f1, 4), "n_pred", scores["n_pred"],
                      flush=True)
                if best is None or f1 > best[1]:
                    best = (penalty, f1, int(scores["n_pred"]))
            row["viterbi"] = {"penalty": best[0], "f1": best[1], "n_pred": best[2]}
            print("  viterbi", best[0], "f1", round(best[1], 4), flush=True)
        rows[held_out] = row

    out = {"cycle_id": "C09-exploratory", "decode": bool(args.decode), "folds": rows}
    path = os.path.join(HERE, "out", "teamlearn_probe.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
