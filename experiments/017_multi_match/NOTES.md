# 017 multi-match validation and ablations (2026-11-08)

All training ran on the marimo/molab sandbox; local runs were inference, decoding and scoring only. Training and held-out data are other matches only: Metrica sample games 1-3, SkillCorner open data (20 matches, MIT), IDSSE/DFL open data (7 Bundesliga matches, CC BY 4.0). Held-out scoring uses grouped 6-fold cross-validation (a match is never in its own training set) and the official `evaluation/possession_metrics.evaluate`. Labels are each dataset's own derivation, not the challenge ground truth.

Adoption rule: gain of at least 0.01 mean spellF1 and wins on a majority of held-out matches. The public 5-minute window of the challenge match is a sanity check only and is never used to choose.

## Adopted

| Change | 23 matches (spellF1) | 7 IDSSE matches (not used to choose) |
|---|---|---|
| Decoder: kappa 0.02 -> 0.05, sliver merge 0.5 s -> 1.0 s | 0.676 -> 0.686, 19/23 wins | 0.563 -> 0.572, 6/7 wins |
| Lambda grid extended downwards (floor 0.25 -> 0.05), because calibration was saturating | included above | included above |

Rate-target scale 1.2 gave a further +0.014 on IDSSE, but only when very short true spells are scored. With true spells under 4 s excluded (closer to the challenge's long spells: public window mean 14.2 s) scale 1.0 is better, so the shipped scale is 1.0.

## Adopted later: per-player set encoder (experiment 020)

The TCN now also sees an attention-and-max pooled encoding of every player (10 features each: attacking-frame position, velocity, speed, distance to the ball and closing speed when the ball is seen, nearest-opponent distance), so individual players carry information beyond the pooled team features. Enabled by `USE_PP=1` in `features.py`; `weights_pp/` holds the shipped 3-seed ensemble trained on all 30 matches (23 plus 7 IDSSE).

| Held-out (30 matches, same decoder) | Standard TCN | + player set | Wins |
|---|---|---|---|
| all spells | 0.661 | 0.679 (+0.018) | 26 of 30 (Metrica 3/3, SkillCorner 16/20, IDSSE 7/7) |
| true spells under 4 s excluded | 0.733 | 0.737 (+0.005) | 19 of 30 |

The gain is concentrated in short spells, as the hypothesis predicted. It passes the adoption rule on all spells and is not negative anywhere when short spells are excluded. Public-window diagnostic only (never a selector): spellF1 0.747, macroF1 0.919, 13 of 17 spells (previous 0.6815, 10 of 17). Adding IDSSE to the training mix also moved the source-balanced possession rate from 2.97 to 3.51 per minute and the dead-ball prior shift from +0.094 to +0.004 (data-driven, no public-window input).

## Shipped weights: no Metrica

The Metrica sample data has no explicit licence, so the shipped ensemble (`weights_nometrica/`) is trained on SkillCorner open data (MIT) and IDSSE/DFL (CC BY 4.0) only: 27 matches, 6-fold held-out scoring on all 30 (Metrica matches are still scored, never trained on in the folds). It matches the full mix: 0.679 vs 0.679 spellF1 (+0.0004, 16 of 30 wins), 0.737 vs 0.737 with true spells under 4 s excluded. Metrica games score 0.674 vs 0.685 (3 matches), a loss within noise. Public-window diagnostic only: see the pipeline output below.

## Tried and not adopted (all below the +0.01 bar)

| Idea | Result |
|---|---|
| Touch-hold ownership override (origin/rasya) on top of our dead-ball model | -0.15 to -0.19 spellF1 (SkillCorner 0.685 -> 0.44); only IDSSE (clean ball) gains, +0.026 |
| Ball-gap interpolation up to 25 frames | -0.0003 |
| Last-touch memory as TCN inputs (held team, time since touch) | +0.0008 |
| Long-context TCN (receptive field about 103 s instead of about 26 s) | +0.004 overall, -0.003 with true spells under 4 s excluded |
| Change-point head as a third output, used as a time-varying switch cost | +0.003 to +0.005, and slightly negative with short spells excluded |
| Adding IDSSE to the 23-match training set | +0.007 (neutral, harmless) |
| 28-config decoder sweep (gate radius, tau, thresholds, shifts, rate) | no single setting clears the bar; kappa = 0 costs 0.33 |

## What the diagnostics say

- Frame-level accuracy is high (team accuracy 0.89 to 0.95, alive/dead F1 about 0.98) but spellF1 is 0.56 to 0.69: the loss is in whole spells.
- Recovery of true spells by duration (a predicted spell of the same team with IoU of at least 0.3): under 2 s about 14 to 23%, 2 to 4 s about 51 to 58%, 4 to 8 s 81%, 8 s and longer 92% or more. Spells under 4 s are 30% of true spells in the 23 matches and 53% in IDSSE.
- With true spells under 4 s excluded, spellF1 is about 0.73 (23 matches) and 0.78 (IDSSE).
- Head-to-head with origin/rasya's decoder on the 7 IDSSE matches (same scorer): ours 0.563 spellF1 / 0.914 macroF1, theirs 0.520 / 0.854. On the public window: ours 0.681, theirs 0.400 after correcting a speed-unit bug in their BAR-ATH loader (0.014 as written).

## Limits

- The ball is missing on 41% of graded frames of the challenge match; recovering it needs the tactical homography, which is not in `data/tactical/`. A self-calibration attempt (018) failed because the camera moves.
- Possession definitions differ between providers (possessions per minute: StatsBomb 1.96, SkillCorner 2.59, Metrica 3.25, public window 3.4, IDSSE 4.3), so the rate target is partly a guess; a 15% error costs less than 0.01 spellF1.
- Metrica sample data has no explicit licence. SkillCorner-only training is within 0.002 of the full mix on SkillCorner matches if Metrica must be dropped.
