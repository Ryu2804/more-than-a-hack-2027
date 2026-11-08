#!/usr/bin/env python3
"""C08: dead cue with a longer slow-run component, on the IDSSE possession harness.

C07 showed that dead recall is already saturated (0.95-0.97) but precision is 0.77-0.79, and the
false dead comes from short runs where the nearest player is momentarily slow. Requiring the slow
condition to hold for 4-6 seconds (100-150 frames) filters those pauses while keeping real stoppages.

Decoder frozen from C05 (r 0.8 m, minlen 30). Merge of short D runs from C06. Slow run, out run and
merge k are selected on the four training matches of each LOSO fold.

Pre-registered in cycles/C08/PREREG_C08_LONGSLOW.md. No window numbers are read or written.
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
from deployable import build_prediction  # noqa: E402

R_M = 0.8
MINLEN = 30
SLOW_MS = 2.0
SLOW_RUNS = (50, 75, 100, 125, 150, 200)
OUT_RUNS = (15, 25)
KS = (0, 50, 100, 150, 250, 400)
BOOTSTRAP = 2000
SEED = 20261108
BASE_SLOW, BASE_OUT = 25, 15


def cue_parts(npz):
    answered, index, speed = P2.nearest(npz, npz["ball_x"], npz["ball_y"])
    outside = answered & ((np.abs(npz["ball_x"]) > P2.PITCH_LENGTH / 2.0 + P2.MARGIN) |
                          (np.abs(npz["ball_y"]) > P2.PITCH_WIDTH / 2.0 + P2.MARGIN))
    slow = answered & np.isfinite(speed) & (speed < SLOW_MS)
    return outside, slow


def cue_of(outside, slow, slow_run, out_run):
    return P2.run_mask(outside, out_run) | P2.run_mask(slow, slow_run)


def quality(npz, dead):
    status = npz["ball_status"]
    valid = status >= 0
    is_dead = valid & (status == 0)
    return {
        "precision": float(is_dead[dead].mean()) if dead.any() else 0.0,
        "recall": float((dead & is_dead).sum() / max(1, is_dead.sum())),
        "flagged_fraction": float(dead.mean()),
    }


def main():
    contexts = {}
    for name in H.DEV:
        context = T.match_context(name)
        npz = H.load(name)
        outside, slow = cue_parts(npz)
        context["npz"] = npz
        context["outside"] = outside
        context["slow"] = slow
        context["held"] = T.hold(context["dmin"], context["team"], R_M, MINLEN)
        context["base_cue"] = cue_of(outside, slow, BASE_SLOW, BASE_OUT)
        context["cues"] = {(sr, orr): cue_of(outside, slow, sr, orr)
                           for sr in SLOW_RUNS for orr in OUT_RUNS}
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
        for slow_run in SLOW_RUNS:
            for out_run in OUT_RUNS:
                for k in KS:
                    values = [score(name, contexts[name]["cues"][(slow_run, out_run)], k)[0]
                              for name in train]
                    table[(slow_run, out_run, k)] = float(np.mean(values))
        best = max(table, key=lambda key: (table[key], key[0], -key[1], -key[2]))
        slow_run, out_run, k = best

        base_table = {}
        for kk in KS:
            base_table[kk] = float(np.mean([score(name, contexts[name]["base_cue"], kk)[0]
                                            for name in train]))
        base_k = max(KS, key=lambda kk: (base_table[kk], -kk))

        oracle_table = {}
        for kk in KS:
            oracle_table[kk] = float(np.mean([score(name, contexts[name]["gt"] == 2, kk)[0]
                                              for name in train]))
        oracle_k = max(KS, key=lambda kk: (oracle_table[kk], -kk))

        context = contexts[held_out]
        dead = context["cues"][(slow_run, out_run)]
        long_f1, long_n = score(held_out, dead, k)
        base_f1, base_n = score(held_out, context["base_cue"], base_k)
        oracle_f1, oracle_n = score(held_out, context["gt"] == 2, oracle_k)

        folds[held_out] = {
            "selected": {"slow_run": slow_run, "out_run": out_run, "k": k,
                         "base_k": base_k, "oracle_k": oracle_k},
            "longslow": {"f1": long_f1, "n_pred": long_n, "dead": quality(context["npz"], dead)},
            "cue": {"f1": base_f1, "n_pred": base_n,
                    "dead": quality(context["npz"], context["base_cue"])},
            "deadoracle": {"f1": oracle_f1, "n_pred": oracle_n},
            "delta": long_f1 - base_f1,
        }
        print(held_out, "slow", slow_run, "out", out_run, "k", k,
              "long", round(long_f1, 4), "cue", round(base_f1, 4),
              "oracle", round(oracle_f1, 4), flush=True)

    deltas = np.array([folds[m]["delta"] for m in H.DEV])
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean()
                      for _ in range(BOOTSTRAP)])
    low, high = np.percentile(means, [5, 95])
    long_mean = float(np.mean([folds[m]["longslow"]["f1"] for m in H.DEV]))
    cue_mean = float(np.mean([folds[m]["cue"]["f1"] for m in H.DEV]))
    oracle_mean = float(np.mean([folds[m]["deadoracle"]["f1"] for m in H.DEV]))
    wins = int((deltas > 0).sum())
    result = {
        "cycle_id": "C08",
        "config": {"r_m": R_M, "minlen": MINLEN, "slow_runs": list(SLOW_RUNS),
                   "out_runs": list(OUT_RUNS), "ks": list(KS),
                   "selection": "slow_run, out_run, k dipilih per fold pada empat match lain"},
        "folds": folds,
        "mean": {
            "touch_hold_longslow_dead_merge": long_mean,
            "touch_hold_cue_dead_merge": cue_mean,
            "touch_hold_deadoracle": oracle_mean,
            "delta": float(deltas.mean()),
            "delta_ci90": [float(low), float(high)],
            "wins": wins,
        },
        "selected": [folds[m]["selected"] for m in H.DEV],
        "supported": bool(deltas.mean() >= 0.04 and low > 0 and wins >= 4),
        "note": "cue out_or_slow2 dengan slow_run panjang; oracle hanya batas atas",
    }
    path = os.path.join(HERE, "out", "deadlong_loso.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("mean long", round(long_mean, 4), "cue", round(cue_mean, 4),
          "oracle", round(oracle_mean, 4), "delta", round(float(deltas.mean()), 4),
          "ci", [round(low, 4), round(high, 4)], "wins", wins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
