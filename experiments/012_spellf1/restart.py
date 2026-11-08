#!/usr/bin/env python3
"""C10: restart seeding after long dead runs.

touch_hold only updates the team at a touch (dmin < 0.8 m). After a long stoppage the next touch can
be late or undetected, so the old team is carried across the restart. This adds an event at the first
alive frame after every dead run of length >= L and takes the nearest-player team there.

Dead is longslow (slow_run 400, out_run 15). Decoder and merge are frozen (r 0.8, minlen 30, k 150).
L is selected per LOSO fold on the other four matches. Pre-registered in cycles/C10.
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
import deadlong as DL  # noqa: E402
import deployable as D  # noqa: E402
import harness as H  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402

R_M, MINLEN, MERGE_K = 0.8, 30, 150
SLOW_RUN, OUT_RUN = 400, 15
LS = (100, 200, 400)
BOOTSTRAP = 2000
SEED = 20261108


def runs(flags):
    return P2.runs_of(flags)


def hold_events(dmin, team, touch_mask, event_frames, r, minlen):
    """Hold team between events. touch_mask marks touch frames; event_frames are extra re-seeds."""
    usable = (dmin < r) & (team >= 0)
    frames = list(np.flatnonzero(usable)) + [int(f) for f in event_frames if 0 <= f < len(team)]
    frames = np.unique(np.array(frames, dtype=np.int64))
    frames = frames[team[frames] >= 0]
    out = np.full(len(team), -1, dtype=np.int8)
    if len(frames):
        values = team[frames].astype(np.int8)
        position = np.searchsorted(frames, np.arange(len(team)), side="right") - 1
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


def restart_frames(dead, min_len):
    """First alive frame after every dead run of length >= min_len."""
    out = []
    for start, stop in runs(dead):
        if stop - start >= min_len:
            frame = stop
            if frame < len(dead) and not dead[frame]:
                out.append(frame)
            elif frame < len(dead):
                while frame < len(dead) and dead[frame]:
                    frame += 1
                if frame < len(dead):
                    out.append(frame)
    return out


def main():
    contexts = {}
    for name in H.DEV:
        ctx = T.match_context(name)
        npz = H.load(name)
        outside, slow = DL.cue_parts(npz)
        ctx["dead"] = DL.cue_of(outside, slow, SLOW_RUN, OUT_RUN)
        ctx["cue"] = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])[0]
        ctx["long"] = T.hold(ctx["dmin"], ctx["team"], R_M, MINLEN)
        contexts[name] = ctx

    def score(ctx, codes):
        pred = D.build_prediction(ctx, ctx["dead"], codes, MERGE_K)
        result = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
        return float(result["mean_f1"]), int(result["n_pred"])

    folds = {}
    for held_out in H.DEV:
        train = [m for m in H.DEV if m != held_out]
        train_scores = {}
        for L in LS:
            values = []
            for name in train:
                ctx = contexts[name]
                frames = restart_frames(ctx["dead"], L)
                codes = hold_events(ctx["dmin"], ctx["team"], ctx["dmin"] < R_M, frames, R_M, MINLEN)
                values.append(score(ctx, codes)[0])
            train_scores[L] = float(np.mean(values))
        best_L = max(LS, key=lambda L: (train_scores[L], L))

        ctx = contexts[held_out]
        frames = restart_frames(ctx["dead"], best_L)
        codes = hold_events(ctx["dmin"], ctx["team"], ctx["dmin"] < R_M, frames, R_M, MINLEN)
        restart_f1, restart_n = score(ctx, codes)
        long_f1, long_n = score(ctx, ctx["long"])
        pred_cue = D.build_prediction(ctx, ctx["cue"], ctx["long"], MERGE_K)
        cue_scores = pm.spell_scores(ctx["gt"], pred_cue, ctx["runs"])
        cue_f1 = float(cue_scores["mean_f1"])

        folds[held_out] = {
            "selected_L": best_L,
            "train_scores": {str(L): train_scores[L] for L in LS},
            "restart": {"f1": restart_f1, "n_pred": restart_n,
                        "n_restart_events": len(frames)},
            "longslow": {"f1": long_f1, "n_pred": long_n},
            "cue": {"f1": cue_f1, "n_pred": int(cue_scores["n_pred"])},
            "delta_vs_cue": restart_f1 - cue_f1,
            "delta_vs_longslow": restart_f1 - long_f1,
        }
        print(held_out, "L", best_L, "restart", round(restart_f1, 4),
              "long", round(long_f1, 4), "cue", round(cue_f1, 4), flush=True)

    deltas = np.array([folds[m]["delta_vs_cue"] for m in H.DEV])
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(deltas, size=len(deltas), replace=True).mean()
                      for _ in range(BOOTSTRAP)])
    low, high = np.percentile(means, [5, 95])
    result = {
        "cycle_id": "C10",
        "config": {"r_m": R_M, "minlen": MINLEN, "merge_k": MERGE_K,
                   "slow_run": SLOW_RUN, "out_run": OUT_RUN, "ls": list(LS)},
        "folds": folds,
        "mean": {
            "touch_hold_restart_longslow": float(np.mean([folds[m]["restart"]["f1"] for m in H.DEV])),
            "touch_hold_longslow": float(np.mean([folds[m]["longslow"]["f1"] for m in H.DEV])),
            "touch_hold_cue": float(np.mean([folds[m]["cue"]["f1"] for m in H.DEV])),
            "delta": float(deltas.mean()),
            "delta_ci90": [float(low), float(high)],
            "wins": int((deltas > 0).sum()),
        },
        "selected_L": [folds[m]["selected_L"] for m in H.DEV],
        "supported": bool(deltas.mean() >= 0.04 and low > 0 and (deltas > 0).sum() >= 4),
    }
    path = os.path.join(HERE, "out", "restart_loso.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("mean", {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in result["mean"].items()}, "L", result["selected_L"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
