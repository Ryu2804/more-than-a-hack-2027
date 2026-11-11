#!/usr/bin/env python3
"""Migrate rasya's deployable pipeline (touch-hold + gated dead cue) to IDSSE per-match predictions."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

WORKTREE = os.environ.get("RASYA_WORKTREE", "/tmp/opencode/rasya-worktree")
sys.path.insert(0, os.path.join(WORKTREE, "experiments", "012_spellf1"))
sys.path.insert(0, os.path.join(WORKTREE, "experiments", "011_ball_value"))
sys.path.insert(0, os.path.join(WORKTREE, "evaluation"))
import harness as H  # noqa: E402
import touch as T  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402

KS = (0, 50, 100, 150, 250, 400)
R_M = 0.8
MINLEN = 30
LABELS = {0: "H", 1: "A", 2: "D", 3: "X"}


def merge_short_dead(series, k):
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


def build_prediction(context, cue, codes, k):
    base = np.where(cue, 2, np.where(codes >= 0, codes, 2)).astype(np.int8)
    pred = merge_short_dead(base, k)
    pred = np.where(context["gt"] == 3, 3, pred).astype(np.int8)
    return pred


def f1_of(context, pred):
    scores = pm.spell_scores(context["gt"], pred, context["runs"])
    return float(scores["mean_f1"])


def write_csv(path, codes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label"])
        for i, c in enumerate(codes):
            w.writerow([i, LABELS[int(c)]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", default=os.path.join(WORKTREE, "..", "..", "data", "external", "idsee", "parsed"))
    # resolve default relative to worktree root's parent? better use absolute
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matches", nargs="*", default=None)
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir).resolve()
    _orig_load = H.load
    H.load = lambda match, parsed_dir=str(parsed_dir): _orig_load(match, parsed_dir)
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred").mkdir(parents=True, exist_ok=True)

    matches = args.matches if args.matches else (H.DEV + H.CONFIRM)
    contexts = {}
    cues = {}
    for name in matches:
        ctx = T.match_context(name)
        npz = H.load(name, str(parsed_dir))
        cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
        ctx["cue"] = cue
        ctx["held"] = T.hold(ctx["dmin"], ctx["team"], R_M, MINLEN)
        contexts[name] = ctx
        cues[name] = cue
        gt = H.gt_codes(npz)
        write_csv(out_dir / "gt" / f"{name}_gt.csv", gt)
        print("loaded", name, "frames", len(gt))

    # Precompute grid f1 for every match
    grid = {}
    for name in matches:
        ctx = contexts[name]
        for k in KS:
            pred = build_prediction(ctx, cues[name], ctx["held"], k)
            grid[(name, k)] = f1_of(ctx, pred)

    for target in matches:
        if target in H.DEV:
            train = [m for m in H.DEV if m != target and m in contexts]
        else:
            train = [m for m in H.DEV if m in contexts]
        best_key = None
        best_mean = -1.0
        for k in KS:
            mean = float(np.mean([grid[(m, k)] for m in train]))
            key = (mean, -k)
            if best_key is None or key > best_key:
                best_key = key
                best_k = k
                best_mean = mean
        ctx = contexts[target]
        pred = build_prediction(ctx, cues[target], ctx["held"], best_k)
        write_csv(out_dir / "pred" / f"{target}_pred.csv", pred)
        print(target, "selected_k", best_k, "train_mean", round(best_mean, 4),
              "pred_spellf1", round(f1_of(ctx, pred), 4))

    print("done; outputs in", out_dir)


if __name__ == "__main__":
    main()