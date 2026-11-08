#!/usr/bin/env python3
"""Starter: touch_hold pipeline on one IDSSE match, with spellF1.

Run:
    /home/rasyaarya/lomba/more_than_a_hack/more-than-a-hack-2027/.venv/bin/python \\
        experiments/012_spellf1/starter_touch.py

What it does (step 4 of the beginner roadmap):
  1. Load one parsed IDSSE match (J03WOH) from ../data/external/idsee/parsed.
  2. Drop referee tracks (they are not players and break the two-team mapping).
  3. Team = nearest player; flip the home/away convention once if agreement < 0.5.
  4. touch_hold: keep the nearest-player team only on "touch" frames (dmin < 0.8 m),
     hold it until the next touch, merge segments shorter than 30 frames.
  5. Dead = cue out_or_slow2 (ball out of pitch for >= 15 frames, or nearest player
     slower than 2.0 m/s for >= 25 frames); merge dead runs shorter than 150 frames.
  6. Build the H/A/D/X prediction, score spellF1 against the provider ground truth,
     and print a per-IoU breakdown.

No training, no model, no submission CSV. ~150k frames, runs in a few seconds.

Next step (stage 3): replace the dead cue with a learned detector trained on Metrica.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPELL = HERE
EVAL = os.path.join(ROOT, "evaluation")
BALL = os.path.join(ROOT, "experiments", "011_ball_value")
sys.path.insert(0, SPELL)
sys.path.insert(0, EVAL)
sys.path.insert(0, BALL)

import harness as H  # noqa: E402
import phase2 as P2  # noqa: E402
import possession_metrics as pm  # noqa: E402
import touch as T  # noqa: E402
from deployable import build_prediction  # noqa: E402

MATCH = "J03WOH"
R_M, MINLEN, MERGE_K = 0.8, 30, 150


def main():
    ctx = T.match_context(MATCH)
    npz = H.load(MATCH)
    cue, _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
    held = T.hold(ctx["dmin"], ctx["team"], R_M, MINLEN)
    pred = build_prediction(ctx, cue, held, MERGE_K)

    scores = pm.spell_scores(ctx["gt"], pred, ctx["runs"])
    print("match:", MATCH)
    print("graded frames:", len(ctx["gt"]), " GT spells:", scores["n_gt"],
          " pred spells:", scores["n_pred"])
    print("team agreement (nearest, alive):", round(H.team_agreement(ctx["gt"], ctx["team"]), 4))
    print("dead cue fires on", round(float(cue.mean()), 3), "of frames")
    print("spellF1 per IoU:", {f"{t:.1f}": round(v["f1"], 4) for t, v in scores["per_iou"].items()})
    print("spellF1 mean (0.3..0.7):", round(scores["mean_f1"], 4))
    print("precision/recall per IoU:", {f"{t:.1f}": (round(v["precision"],3), round(v["recall"],3))
                                         for t,v in scores["per_iou"].items()})

    # Optional: write a per-frame CSV for inspection (not a submission).
    out = os.path.join(SPELL, "starter_touch_out.csv")
    import numpy as np
    labels = np.array(["H", "A", "D", "X"])[pred]
    with open(out, "w") as f:
        f.write("frame,label\n")
        for i, lab in enumerate(labels):
            f.write(f"{i},{lab}\n")
    print("wrote", out, "(single-match inspection file, not a submission)")


if __name__ == "__main__":
    main()
