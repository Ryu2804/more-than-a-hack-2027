#!/usr/bin/env python3
"""C07: dead from player speed (stoppage detector) on the IDSSE possession harness.

C06 showed that the dead cue, not the possession decoder, is the bottleneck: out_or_slow2 fires on
~21% of frames with P(dead|cue) 0.32, and the false D runs cut possessions. When play stops, most
players stand still, so the median player speed is a high-precision dead signal.

Decoder is frozen from C05 (r 0.8 m, minlen 30). Dead runs shorter than k frames are merged into the
previous segment (C06). Threshold t, minimum run m, and merge k are selected on the four training
matches of each LOSO fold.

Pre-registered in cycles/C07/PREREG_C07_DEADSPEED.md. No window numbers are read or written.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
import harness as H  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402
from deployable import build_prediction, merge_short_dead  # noqa: E402

R_M = 0.8
MINLEN = 30
THRESHOLDS = (0.6, 0.8, 1.0)
MINRUNS = (1, 25)
KS = (0, 50, 100, 150, 250, 400)
BOOTSTRAP = 2000
SEED = 20261108


def min_run(flags, minlen):
    out = np.zeros(len(flags), dtype=bool)
    if minlen <= 1:
        return flags.astype(bool)
    starts = np.flatnonzero(np.r_[True, flags[1:] != flags[:-1]])
    ends = np.r_[starts[1:], len(flags)]
    for start, stop in zip(starts, ends):
        if flags[start] and stop - start >= minlen:
            out[start:stop] = True
    return out


def median_player_speed(npz):
    speeds = np.where(npz["player_mask"], npz["player_speed"].astype(np.float64) / 3.6, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(speeds, axis=0)


def dead_cue(npz, threshold, minlen):
    median = median_player_speed(npz)
    flag = np.isfinite(median) & (median < threshold)
    return min_run(flag, minlen)


def dead_quality(npz, cue):
    status = npz["ball_status"]
    valid = status >= 0
    dead = valid & (status == 0)
    if not cue.any():
        return {"precision": 0.0, "recall": 0.0, "flagged_fraction": 0.0}
    return {
        "precision": float(dead[cue].mean()),
        "recall": float((cue & dead).sum() / max(1, dead.sum())),
        "flagged_fraction": float(cue.mean()),
    }


def main():
    contexts = {}
    for name in H.DEV:
        context = T.match_context(name)
        npz = H.load(name)
        cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
        context["cue"] = cue
        context["held"] = T.hold(context["dmin"], context["team"], R_M, MINLEN)
        context["npz"] = npz
        contexts[name] = context

    def score(name, dead, k):
        context = contexts[name]
        pred = build_prediction(context, dead, context["held"], k)
        result = pm.spell_scores(context["gt"], pred, context["runs"])
        return float(result["mean_f1"]), int(result["n_pred"])

    folds = {}
    for held_out in H.DEV:
        train = [m for m in H.DEV if m != held_out]

        table = {}
        for threshold in THRESHOLDS:
            for minlen in MINRUNS:
                for k in KS:
                    values = []
                    for name in train:
                        dead = dead_cue(contexts[name]["npz"], threshold, minlen)
                        values.append(score(name, dead, k)[0])
                    table[(threshold, minlen, k)] = float(np.mean(values))
        best = max(table, key=lambda key: (table[key], key[0], key[1], -key[2]))
        threshold, minlen, k = best

        train_cue = {}
        for kk in KS:
            train_cue[kk] = float(np.mean([score(name, contexts[name]["cue"], kk)[0]
                                           for name in train]))
        best_cue_k = max(KS, key=lambda kk: (train_cue[kk], -kk))

        train_oracle = {}
        for kk in KS:
            train_oracle[kk] = float(np.mean([1.0 * score(name, contexts[name]["gt"] == 2, kk)[0]
                                              for name in train]))
        best_oracle_k = max(KS, key=lambda kk: (train_oracle[kk], -kk))

        context = contexts[held_out]
        dead = dead_cue(context["npz"], threshold, minlen)
        medp_f1, medp_n = score(held_out, dead, k)
        cue_f1, cue_n = score(held_out, context["cue"], best_cue_k)
        oracle_f1, oracle_n = score(held_out, context["gt"] == 2, best_oracle_k)

        folds[held_out] = {
            "selected": {"threshold": threshold, "minlen": minlen, "k": k,
                         "cue_k": best_cue_k, "oracle_k": best_oracle_k},
            "medpspeed": {"f1": medp_f1, "n_pred": medp_n,
                          "dead": dead_quality(context["npz"], dead)},
            "cue": {"f1": cue_f1, "n_pred": cue_n},
            "deadoracle": {"f1": oracle_f1, "n_pred": oracle_n},
            "delta": medp_f1 - cue_f1,
            "train_best": table[best],
        }
        print(held_out, "t", threshold, "m", minlen, "k", k,
              "medp", round(medp_f1, 4), "cue", round(cue_f1, 4),
              "oracle", round(oracle_f1, 4), flush=True)

    deltas = np.array([folds[m]["delta"] for m in H.DEV])
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean()
                      for _ in range(BOOTSTRAP)])
    low, high = np.percentile(means, [5, 95])
    medp_mean = float(np.mean([folds[m]["medpspeed"]["f1"] for m in H.DEV]))
    cue_mean = float(np.mean([folds[m]["cue"]["f1"] for m in H.DEV]))
    oracle_mean = float(np.mean([folds[m]["deadoracle"]["f1"] for m in H.DEV]))
    wins = int((deltas > 0).sum())
    result = {
        "cycle_id": "C07",
        "config": {"r_m": R_M, "minlen": MINLEN, "thresholds": list(THRESHOLDS),
                   "minruns": list(MINRUNS), "ks": list(KS),
                   "selection": "t, m, k dipilih per fold pada empat match lain"},
        "folds": folds,
        "mean": {
            "touch_hold_medpspeed_dead_merge": medp_mean,
            "touch_hold_cue_dead_merge": cue_mean,
            "touch_hold_deadoracle": oracle_mean,
            "delta": float(deltas.mean()),
            "delta_ci90": [float(low), float(high)],
            "wins": wins,
        },
        "selected": [folds[m]["selected"] for m in H.DEV],
        "supported": bool(deltas.mean() >= 0.04 and low > 0 and wins >= 4),
        "note": "median kecepatan pemain sebagai dead; cue out_or_slow2 dan oracle hanya pembanding",
    }
    path = os.path.join(HERE, "out", "deadspeed_loso.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("mean medp", round(medp_mean, 4), "cue", round(cue_mean, 4),
          "oracle", round(oracle_mean, 4), "delta", round(float(deltas.mean()), 4),
          "ci", [round(low, 4), round(high, 4)], "wins", wins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
