#!/usr/bin/env python3
"""C06: deployable dead (cue) + touch hold, IDSSE leave-one-match-out.

C05 showed touch_hold beats nearest_smooth50 when dead is oracle. This cycle replaces oracle dead
with the out_or_slow2 cue and merges short D runs back into the surrounding possession. The merge
threshold k is selected on the four training matches of each fold; r = 0.8 m and minlen = 30 are
frozen from C05.

Pre-registered in cycles/C06/PREREG_C06_DEPLOYABLE.md. No window numbers are read or written.
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

R_M = 0.8
MINLEN = 30
KS = (0, 50, 100, 150, 250, 400)
BOOTSTRAP = 2000
SEED = 20261108


def merge_short_dead(series, k):
    """Replace D runs shorter than k frames by the previous segment value."""
    out = series.copy()
    if k <= 0:
        return out
    n = len(out)
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        if out[a] == 2 and (b - a) < k:
            value = out[a - 1] if a > 0 else (out[b] if b < n else 2)
            out[a:b] = value
    return out


def build_prediction(context, cue, codes, k, oracle_dead=False):
    if oracle_dead:
        base = np.where(context["gt"] == 2, 2, np.where(codes >= 0, codes, 2)).astype(np.int8)
    else:
        base = np.where(cue, 2, np.where(codes >= 0, codes, 2)).astype(np.int8)
    pred = merge_short_dead(base, k) if not oracle_dead else base
    pred = np.where(context["gt"] == 3, 3, pred).astype(np.int8)
    return pred


def f1_of(context, pred):
    scores = pm.spell_scores(context["gt"], pred, context["runs"])
    return float(scores["mean_f1"]), int(scores["n_pred"])


def bootstrap_delta(values, reps=BOOTSTRAP, seed=SEED):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(reps)])
    low, high = np.percentile(means, [5, 95])
    return float(values.mean()), float(low), float(high)


def main():
    contexts = {}
    for name in H.DEV:
        context = T.match_context(name)
        npz = H.load(name)
        cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
        context["cue"] = cue
        context["smooth50"] = H.smooth_team(context["team"], 50)
        context["held"] = T.hold(context["dmin"], context["team"], R_M, MINLEN)
        contexts[name] = context

    folds = {}
    for held_out in H.DEV:
        train = [m for m in H.DEV if m != held_out]

        def mean_for(k, arm):
            values = []
            for name in train:
                context = contexts[name]
                codes = context["held"] if arm == "touch" else context["smooth50"]
                pred = build_prediction(context, context["cue"], codes, k)
                values.append(f1_of(context, pred)[0])
            return float(np.mean(values))

        scores = {k: (mean_for(k, "touch"), mean_for(k, "nearest")) for k in KS}
        best_k = max(KS, key=lambda k: (scores[k][0], -k))

        context = contexts[held_out]
        row = {"selected_k": best_k, "train_scores": {str(k): scores[k] for k in KS}}
        for arm, codes in (("touch_hold", context["held"]),
                           ("nearest_smooth50", context["smooth50"])):
            pred = build_prediction(context, context["cue"], codes, best_k)
            f1, n_pred = f1_of(context, pred)
            row[arm + "_deadcue"] = {"f1": f1, "n_pred": n_pred}
            oracle = build_prediction(context, context["cue"], codes, 0, oracle_dead=True)
            f1o, n_predo = f1_of(context, oracle)
            row[arm + "_deadoracle"] = {"f1": f1o, "n_pred": n_predo}
        row["delta"] = row["touch_hold_deadcue"]["f1"] - row["nearest_smooth50_deadcue"]["f1"]
        folds[held_out] = row
        print(held_out, "k", best_k, "touch", round(row["touch_hold_deadcue"]["f1"], 4),
              "nearest", round(row["nearest_smooth50_deadcue"]["f1"], 4),
              "delta", round(row["delta"], 4), flush=True)

    deltas = np.array([folds[m]["delta"] for m in H.DEV])
    mean, low, high = bootstrap_delta(deltas)
    touch_mean = float(np.mean([folds[m]["touch_hold_deadcue"]["f1"] for m in H.DEV]))
    nearest_mean = float(np.mean([folds[m]["nearest_smooth50_deadcue"]["f1"] for m in H.DEV]))
    oracle_touch = float(np.mean([folds[m]["touch_hold_deadoracle"]["f1"] for m in H.DEV]))
    oracle_nearest = float(np.mean([folds[m]["nearest_smooth50_deadoracle"]["f1"] for m in H.DEV]))
    wins = int((deltas > 0).sum())
    selected = [folds[m]["selected_k"] for m in H.DEV]
    result = {
        "cycle_id": "C06",
        "config": {"r_m": R_M, "minlen": MINLEN, "ks": list(KS),
                   "selection": "k dipilih per fold pada empat match lain"},
        "folds": folds,
        "mean": {
            "touch_hold_deadcue": touch_mean,
            "nearest_smooth50_deadcue": nearest_mean,
            "touch_hold_deadoracle": oracle_touch,
            "nearest_smooth50_deadoracle": oracle_nearest,
            "delta": mean,
            "delta_ci90": [low, high],
            "wins": wins,
        },
        "selected_k": selected,
        "supported": bool(mean >= 0.04 and low > 0 and wins >= 4),
        "note": "dead diprediksi cue out_or_slow2; oracle dead hanya referensi batas atas",
    }
    path = os.path.join(HERE, "out", "deployable_loso.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("mean touch", round(touch_mean, 4), "nearest", round(nearest_mean, 4),
          "delta", round(mean, 4), "ci", [round(low, 4), round(high, 4)],
          "wins", wins, "k", selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
