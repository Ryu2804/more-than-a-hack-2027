# 004 - Jersey-colour check of the ATD team labels

## Hypothesis

The tracking says which team each track slot belongs to, but slots are not persons. If the
jersey colour of a slot drifts to the other team's colour for a sustained stretch, the slot has
changed player (or the static team label is wrong), and that is the source of the 28 team
switches per minute seen in 002/003. The video can measure this without a detector and without
any hand label: project the ATD position through the homography, crop the torso, compare the
colour with the two team centroids that the ATD labels themselves define.

## Assets used

- `data/tactical/BAR-ATH_tactical.mp4` (199,319 frames, 25 fps, 1080p)
- `data/tactical/BAR-ATH_tactical_atd.txt` / `_atd.xml` (positions and team per slot)
- `data/tactical/BAR-ATH_tactical_homography.csv` (pitch -> pixels, 100% valid)

Nothing from ground truth. No frame inspected by a person; features are computed for every
sampled frame.

## Change vs previous

002/003 only used positions. This is the first experiment that decodes the video.

## Run

```
.venv/bin/python experiments/004_video_team_check/run.py \
  --data-dir ../data --out-dir out/004_video_team_check --step 5 --save-visuals
```

(5-frame step, every fifth graded frame; a full-frame pass costs about 5x more for little extra
signal at this stage.)

## Result

| metric | value |
|---|---|
| sampled graded frames | 31,772 (30,907 with usable torso boxes) |
| colour agreement with ATD team | mean 0.880, median 0.889, p10 0.769 |
| Fisher separability (median) | 6.28 |
| team centroids (blue, red, white) | home [0.033, 0.022, 0.070], away [0.012, 0.002, 0.504] |

Kits separate almost entirely on the white fraction: Barcelona (blue/red) has 0.03-0.10 white,
Athletic (white/red) 0.35-0.65 on every slot.

Longest sustained colour disagreement with the ATD team, per slot, in seconds:
34 -> 55.0, 44 -> 49.2, 37 -> 32.0, 24 -> 27.6, 29 -> 24.4, 16 -> 20.6, 30 -> 20.2, 35 -> 19.4.

## Verdict

Exploratory, mechanism plausible (bukan "supported" — tidak ada pra-registrasi): the video does
carry a strong, label-free team signal, and a large fraction of slot-time is colour-inconsistent
with the slot's static team. That is consistent with slot reuse / identity switches rather than
with a broken homography (projections land on players).

Not yet shown: that repairing the label with this signal improves spellF1. That is 005.
