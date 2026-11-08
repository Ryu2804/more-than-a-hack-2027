#!/usr/bin/env python3
"""Starter, step 2: run the touch_hold pipeline on ALL 5 IDSSE dev matches.

This is the smallest honest validation: instead of trusting one match (0.5294),
run the same recipe on 5 matches and look at the spread. If the mean is ~0.52 and
every match is in [0.50, 0.57], the pipeline is stable; if one match is an
outlier, investigate before doing anything else.

What it does (no model, no training, ~5 seconds):
  for each match in DEV:
    - load parsed npz (IDSSE, an external league with full labels)
    - drop referee tracks (they are not players)
    - nearest-player team, flipped once per match if agreement < 0.5
    - touch_hold decoder (r=0.8 m, minlen 30)
    - dead cue = out_or_slow2, merge D runs < 150 frames
    - score spellF1 (mean over IoU 0.3..0.7)
  print per-match + mean + min/max, and write a small JSON.

This is NOT a submission. Submission needs the BAR-ATH game (gated download) and
validate_submission.py. See DATA.md.

Run:
    .venv/bin/python experiments/012_spellf1/starter_touch_all.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "011_ball_value"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np

import harness as H
import phase2 as P2
import possession_metrics as pm
import touch as T
from deployable import build_prediction

DEV = H.DEV  # [J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ]


def run_one(match):
    ctx = T.match_context(match)
    npz = H.load(match)
    cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
    held = T.hold(ctx["dmin"], ctx["team"], 0.8, 30)
    pred = build_prediction(ctx, cue, held, 150)
    scores = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
    return {
        "match": match,
        "gt_spells": int(scores["n_gt"]),
        "pred_spells": int(scores["n_pred"]),
        "team_agreement": float(H.team_agreement(ctx["gt"], ctx["team"])),
        "dead_cue_fraction": float(cue.mean()),
        "spell_f1": float(scores["mean_f1"]),
        "per_iou": {str(k): (v["f1"] if isinstance(v, dict) else float(v))
                    for k, v in scores["per_iou"].items()},
    }


def main():
    results = [run_one(m) for m in DEV]
    f1s = [r["spell_f1"] for r in results]

    print("\n=== touch_hold + cue dead + merge 150, LOSO-style (no model) ===")
    print(f"{'match':8s} {'gt':>4s} {'pred':>5s} {'agree':>6s} {'dead%':>6s} {'F1':>7s}")
    for r in results:
        print(f"{r['match']:8s} {r['gt_spells']:4d} {r['pred_spells']:5d} "
              f"{r['team_agreement']:6.3f} {r['dead_cue_fraction']:6.3f} {r['spell_f1']:7.4f}")
    print(f"{'MEAN':8s} {'':4s} {'':5s} {'':6s} {'':6s} {np.mean(f1s):7.4f}")
    print(f"min {min(f1s):.4f}  max {max(f1s):.4f}  spread {max(f1s)-min(f1s):.4f}")

    out_path = os.path.join(HERE, "starter_touch_all_out.json")
    with open(out_path, "w") as f:
        json.dump({"results": results,
                   "mean_spell_f1": float(np.mean(f1s)),
                   "min": float(min(f1s)), "max": float(max(f1s))}, f, indent=2)
    print(f"\nwrote {out_path}")
    print("baseline organiser (smooth50, full match) = 0.2433")
    print("our pipeline on IDSSE dev = {:.4f}  ({:.2f}x baseline)".format(
        np.mean(f1s), np.mean(f1s) / 0.2433))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY