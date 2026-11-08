#!/usr/bin/env python3
"""C05: last-touch hold decoder on the IDSSE possession harness, leave-one-match-out.

Possession is held on the team of the nearest player while the ball is within r metres, until the
next touch. Segments shorter than `minlen` are merged into the previous segment. Dead is oracle (GT),
as in the C04 harness. Grid selection happens on the other four matches of each fold, so the held-out
match never picks its own configuration.

Pre-registered in cycles/C05/PREREG_C05_TOUCH.md. No window numbers are read or written.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import harness as H  # noqa: E402
import possession_metrics as pm  # noqa: E402

RS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MINLENS = (0, 15, 30, 60, 90)
BOOTSTRAP = 2000
SEED = 20261108


def match_context(name):
    npz = H.load(name)
    gt = H.gt_codes(npz)
    bx, by = npz["ball_x"], npz["ball_y"]
    px, py, played = npz["player_x"], npz["player_y"], npz["player_mask"]
    side = H.sides(npz)
    distance = np.hypot(px - bx[None, :], py - by[None, :])
    finite = np.isfinite(bx) & np.isfinite(by)
    distance = np.where(played & finite[None, :], distance, np.inf)
    index = distance.argmin(axis=0)
    dmin = distance[index, np.arange(distance.shape[1])]
    team = np.where(np.isfinite(dmin), side[index], -1).astype(np.int8)
    flipped = False
    if H.team_agreement(gt, team) < 0.5:
        team = np.where(team >= 0, 1 - team, -1).astype(np.int8)
        flipped = True
    return {
        "gt": gt,
        "runs": pm.valid_runs(gt),
        "dmin": dmin,
        "team": team,
        "flipped": flipped,
        "frames": len(gt),
    }


def hold(dmin, team, r, minlen):
    touch = (dmin < r) & (team >= 0)
    index = np.flatnonzero(touch)
    out = np.full(len(team), -1, dtype=np.int8)
    if len(index):
        values = team[index]
        position = np.searchsorted(index, np.arange(len(team)), side="right") - 1
        out = np.where(position >= 0, values[np.clip(position, 0, len(values) - 1)], -1)
        out = out.astype(np.int8)
    if minlen:
        starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
        ends = np.r_[starts[1:], len(out)]
        for a, b in zip(starts, ends):
            if b - a < minlen:
                if a > 0:
                    out[a:b] = out[a - 1]
                elif b < len(out):
                    out[a:b] = out[b]
    return out


def as_codes(gt, held):
    return np.where(gt == 2, 2, np.where(held == 0, 0, np.where(held == 1, 1, 2)))


def f1_of(context, held):
    scores = pm.spell_scores(context["gt"], as_codes(context["gt"], held), context["runs"])
    return float(scores["mean_f1"]), int(scores["n_pred"])


def bootstrap_delta(values, reps=BOOTSTRAP, seed=SEED):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(reps)])
    low, high = np.percentile(means, [5, 95])
    return float(values.mean()), float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(HERE, "out", "touch_loso.json"))
    args = parser.parse_args()

    contexts, baseline, grid_f1 = {}, {}, {}
    for name in H.DEV:
        context = match_context(name)
        contexts[name] = context
        smoothed = H.smooth_team(context["team"], 50)
        baseline[name] = f1_of(context, smoothed)[0]
        for r in RS:
            for minlen in MINLENS:
                held = hold(context["dmin"], context["team"], r, minlen)
                grid_f1[(name, r, minlen)] = f1_of(context, held)[0]
        print(name, "baseline", round(baseline[name], 4), flush=True)

    folds, deltas, wins = {}, [], 0
    for held_out in H.DEV:
        train = [m for m in H.DEV if m != held_out]
        best_mean, best_key = -1.0, None
        for r in RS:
            for minlen in MINLENS:
                mean = float(np.mean([grid_f1[(m, r, minlen)] for m in train]))
                key = (mean, r, minlen)
                if best_key is None or key > best_key:
                    best_mean, best_key = mean, key
        _, r, minlen = best_key
        touch = grid_f1[(held_out, r, minlen)]
        delta = touch - baseline[held_out]
        deltas.append(delta)
        wins += int(touch > baseline[held_out])
        folds[held_out] = {
            "r": r, "minlen": minlen, "train_mean": best_mean,
            "touch_f1": touch, "baseline_f1": baseline[held_out], "delta": delta,
        }

    mean, low, high = bootstrap_delta(deltas)
    fixed = {}
    for r, minlen in ((0.7, 15), (0.8, 30)):
        values = [grid_f1[(m, r, minlen)] for m in H.DEV]
        fixed["r{:g}_m{}".format(r, minlen)] = {
            "mean": float(np.mean(values)), "per_match": dict(zip(H.DEV, values)),
        }
    best_all = max(((float(np.mean([grid_f1[(m, r, ml)] for m in H.DEV])), r, ml)
                    for r in RS for ml in MINLENS))
    result = {
        "cycle": "C05",
        "split": "dev 5 match, LOSO",
        "bootstrap": BOOTSTRAP,
        "folds": folds,
        "loso_mean": float(np.mean([f["touch_f1"] for f in folds.values()])),
        "baseline_mean": float(np.mean([baseline[m] for m in H.DEV])),
        "delta_mean": mean,
        "delta_ci90": [low, high],
        "wins": wins,
        "baseline_per_match": baseline,
        "fixed_configs": fixed,
        "best_on_all_dev_optimistic": {
            "mean": best_all[0], "r": best_all[1], "minlen": best_all[2],
        },
        "runtime_note": "no public numbers read",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in ("loso_mean", "baseline_mean", "delta_mean",
                                             "delta_ci90", "wins")}, indent=2))
    for name, fold in folds.items():
        print(name, "r", fold["r"], "minlen", fold["minlen"],
              "touch", round(fold["touch_f1"], 4), "base", round(fold["baseline_f1"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
