#!/usr/bin/env python3
"""Skeleton: ball-free dead detector (logistic regression, LOSO) on IDSSE.

Goal: replace the ball-based cue out_or_slow2 with a detector that uses only player collective
motion, so it can also work on BAR-ATH where the ball is frequently missing.

Features (all ball-free, per frame):
  1 median player speed (m/s)
  2 mean player speed
  3 std player speed
  4 fraction of visible players with speed < 0.5 m/s
  5 fraction of visible players with speed > 3.0 m/s
  6 number of visible players
  7 median speed, home team
  8 median speed, away team
  9 abs(median home - median away)
 10 rolling mean of feature 1 over 25 frames
 11 rolling mean of feature 1 over 100 frames

Target: dead (BallStatus == 0) on graded frames (BallStatus >= 0). Train on 4 matches, test on 1
(LOSO). Report frame precision/recall/F1 and spellF1 when plugged into touch_hold + merge 150,
compared with cue out_or_slow2 and medp<1.0. No submission, no window numbers.
"""

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
SEED = 20261109


def player_speeds(npz):
    speeds = np.where(npz["player_mask"], npz["player_speed"].astype(np.float64) / 3.6, np.nan)
    sides = H.sides(npz)
    home = sides == 0
    away = sides == 1
    return speeds, home, away


def features(npz):
    speeds, home, away = player_speeds(npz)
    n_players = npz["player_mask"].sum(axis=0).astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(speeds, axis=0)
        mean = np.nanmean(speeds, axis=0)
        std = np.nanstd(speeds, axis=0)
        frac_slow = (speeds < 0.5).sum(axis=0) / np.maximum(1, n_players)
        frac_fast = (speeds > 3.0).sum(axis=0) / np.maximum(1, n_players)
        med_home = np.nanmedian(np.where(home[:, None], speeds, np.nan), axis=0)
        med_away = np.nanmedian(np.where(away[:, None], speeds, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    med_home = np.where(np.isfinite(med_home), med_home, 0.0)
    med_away = np.where(np.isfinite(med_away), med_away, 0.0)
    diff = np.abs(med_home - med_away)
    # rolling means of median speed
    def roll(x, w):
        out = np.zeros_like(x)
        c = np.cumsum(np.r_[0.0, x])
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            out[i] = (c[i + 1] - c[lo]) / (i - lo + 1)
        return out
    roll25 = roll(med, 25)
    roll100 = roll(med, 100)
    X = np.column_stack([med, mean, std, frac_slow, frac_fast, n_players,
                         med_home, med_away, diff, roll25, roll100])
    return X, med


def train_logreg(X, y, l2=1e-2):
    n, d = X.shape
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / std
    # add bias
    Xb = np.column_stack([np.ones(n), Xs])
    def loss(w):
        z = Xb @ w
        # stable log-sum-exp
        logp = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
        # log likelihood of y=1
        ll = y * logp + (1 - y) * (logp - z)
        reg = 0.5 * l2 * np.sum(w[1:] ** 2)
        return -ll.mean() + reg / n
    def grad(w):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n
        g[1:] += l2 * w[1:] / n
        return g
    w0 = np.zeros(d + 1)
    res = minimize(loss, w0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-9})
    return res.x, mean, std


def predict(w, mean, std, X):
    Xs = (X - mean) / std
    Xb = np.column_stack([np.ones(len(X)), Xs])
    z = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def dead_quality(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "flagged": float(y_pred.mean())}


def main():
    contexts = {}
    for name in H.DEV:
        ctx = T.match_context(name)
        npz = H.load(name)
        X, med = features(npz)
        status = npz["ball_status"]
        graded = status >= 0
        y = (status == 0).astype(np.int64)  # dead=1
        ctx["X"] = X
        ctx["y"] = y
        ctx["graded"] = graded
        ctx["npz"] = npz
        ctx["cue"] = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])[0]
        ctx["long"] = T.hold(ctx["dmin"], ctx["team"], R_M, MINLEN)
        ctx["med"] = med
        contexts[name] = ctx

    def score_spell(ctx, dead_mask):
        pred = build_prediction(ctx, dead_mask, ctx["long"], MERGE_K)
        s = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
        return float(s["mean_f1"]), int(s["n_pred"])

    folds = {}
    for held_out in H.DEV:
        train = [m for m in H.DEV if m != held_out]
        Xtr = np.vstack([contexts[m]["X"][contexts[m]["graded"]] for m in train])
        ytr = np.concatenate([contexts[m]["y"][contexts[m]["graded"]] for m in train])
        w, mean, std = train_logreg(Xtr, ytr, l2=1e-2)

        ctx = contexts[held_out]
        g = ctx["graded"]
        p = predict(w, mean, std, ctx["X"])
        dead_pred = (p > 0.5) & g  # only predict dead on graded frames
        # frame metrics on graded frames
        yhat = dead_pred[g].astype(np.int64)
        ytrue = ctx["y"][g]
        q = dead_quality(ytrue, yhat)
        # dead masks for scoring (full length, False on ungraded)
        dead_mask = np.zeros(len(ctx["gt"]), dtype=bool)
        dead_mask[g] = dead_pred[g]

        # comparators
        medp_dead = (ctx["med"] < 1.0) & g
        cue_dead = ctx["cue"] & g

        bf_f1, bf_n = score_spell(ctx, dead_mask)
        medp_f1, medp_n = score_spell(ctx, medp_dead)
        cue_f1, cue_n = score_spell(ctx, cue_dead)
        oracle_f1, oracle_n = score_spell(ctx, (ctx["npz"]["ball_status"] == 0) & g)

        folds[held_out] = {
            "ballfree": {"f1": bf_f1, "n_pred": bf_n, "dead": q,
                         "frame_acc": float((yhat == ytrue).mean()),
                         "p_mean": float(p[g].mean()), "p_std": float(p[g].std())},
            "medp_lt1": {"f1": medp_f1, "n_pred": medp_n},
            "cue": {"f1": cue_f1, "n_pred": cue_n},
            "oracle": {"f1": oracle_f1, "n_pred": oracle_n},
        }
        print(f"{held_out}: ballfree F1={bf_f1:.4f} (acc {q and folds[held_out]['ballfree']['frame_acc']:.3f}, "
              f"P {q['precision']:.3f} R {q['recall']:.3f}) | medp {medp_f1:.4f} | cue {cue_f1:.4f} | "
              f"oracle {oracle_f1:.4f}", flush=True)

    bf_mean = float(np.mean([folds[m]["ballfree"]["f1"] for m in H.DEV]))
    medp_mean = float(np.mean([folds[m]["medp_lt1"]["f1"] for m in H.DEV]))
    cue_mean = float(np.mean([folds[m]["cue"]["f1"] for m in H.DEV]))
    oracle_mean = float(np.mean([folds[m]["oracle"]["f1"] for m in H.DEV]))
    acc_mean = float(np.mean([folds[m]["ballfree"]["frame_acc"] for m in H.DEV]))
    prec_mean = float(np.mean([folds[m]["ballfree"]["dead"]["precision"] for m in H.DEV]))
    rec_mean = float(np.mean([folds[m]["ballfree"]["dead"]["recall"] for m in H.DEV]))

    result = {
        "cycle": "ballfree_dead_skeleton",
        "features": ["median_speed", "mean_speed", "std_speed", "frac_slow_lt0.5",
                     "frac_fast_gt3", "n_visible", "med_home", "med_away", "abs_diff",
                     "roll25", "roll100"],
        "folds": folds,
        "mean": {
            "ballfree": bf_mean,
            "medp_lt1": medp_mean,
            "cue_out_or_slow2": cue_mean,
            "oracle_dead": oracle_mean,
            "frame_acc": acc_mean,
            "dead_precision": prec_mean,
            "dead_recall": rec_mean,
        },
        "note": "ball-free logistic (no ball position, no BallPossession). Compare with cue (ball-based) "
                "and medp<1.0 (ball-free single threshold).",
    }
    path = os.path.join(HERE, "out", "ballfree_dead.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("\nMEAN: ballfree", round(bf_mean, 4), "| medp<1", round(medp_mean, 4),
          "| cue", round(cue_mean, 4), "| oracle", round(oracle_mean, 4))
    print("frame acc", round(acc_mean, 4), "dead P", round(prec_mean, 4), "R", round(rec_mean, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
