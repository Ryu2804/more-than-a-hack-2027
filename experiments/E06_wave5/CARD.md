# E06: Wave5, short-possession loss (FINAL METHOD)

| | |
|---|---|
| **Folder** | [`experiments/E06_wave5/`](./) |
| **Builds on** | E05 (control `old_gru25/wave4`, dev 0.656490) |
| **Status** | done |
| **Verdict** | **promoted; final team method** |

## Question
Can a training-side change aimed at short possessions (loss weighting, boundary head, or heavier ball masking) beat the wave4 control under the same protocol?

## Hypothesis
Under-counting short possessions is the dominant error. Up-weighting the team-state loss on complete live spells < 4 s raises short-spell recall without hurting long spells.

## Design
- **Three single-factor interventions** on the wave4 PP-TCN, same recipe (1,500 updates/seed, batch 16, crop 384, AdamW 0.002/0.01, OneCycle, dropout 0.15, seeds 0/1/2):
  `boundary_tcn` (auxiliary boundary head), `short_tcn` (×2 state loss on complete live spells < 4 s), `masked_tcn` (full-ball mask 0.6 vs 0.3).
- **Factorial grid:** 12 probability families × 8 decoders = 96 configurations, 2,208 development evaluations, 63 seed fits.
- **Split, folds, aggregation:** identical to E05 (19 training matches, six grouped folds, 23 scenarios / 21 matches).
- **Post-freeze:** holdout (2 IDSSE, 6 SkillCorner, Metrica 3 in two conditions) and a paired ball-drop stress (same 6 SkillCorner matches, 35% of visible-ball samples removed in 10–60-sample spans, seed 20261113).

## Metric and decision rule
Equal-provider mean spellF1. Promote a challenger only if it gains ≥ 0.01 over the control, wins ≥ 2 of 3 providers, and no provider drops > 0.03. Rule hashed before fitting.

## Result
6 of 96 configurations pass. Best: `short_gru25/dead45` = 75% short-loss PP-TCN + 25% frozen wave3 GRU; dead threshold 0.45, minimum dead 1 s, Viterbi reset per live interval, post-merge rate calibration.

| Provider (dev) | Control wave4 | Wave5 | Δ |
|---|---:|---:|---:|
| IDSSE | 0.583755 | 0.612109 | +0.028354 |
| SkillCorner | 0.709044 | 0.709337 | +0.000293 |
| Metrica | 0.676673 | 0.688852 | +0.012179 |
| Equal mean | 0.656490 | 0.670099 | +0.013609 |

Holdout Δ: IDSSE +0.022893, SkillCorner clean +0.002100, SkillCorner stress −0.001025 (wins 2/6), Metrica 3 +0.032838.
Short-spell (< 4 s) recall at IoU 0.75 rises in every cohort (e.g. IDSSE 55/360 → 64/360). Under stress, precision at IoU 0.75 drops 46.13% → 43.84% and recall of spells ≥ 15 s drops 71.98% → 68.64%.
Post-freeze public diagnostic: spellF1 0.772414, macroF1 0.902059 (same spellF1 as wave4).

## Verdict
**Promoted.** Locked as the team's final method (`farras_wave5_short_gru25_dead45`). Frozen CSV SHA-256 `cd59fb90…`, independently rebuilt byte-for-byte (199,319 rows).

## Threats to validity
- All holdout matches previously exposed; stress cohort is a transformation of holdout matches, not new units.
- The decoder changed together with the loss, so the gain is a whole-pipeline effect.
- Worst development count error still 41.88% (> 30%). Short recall < 25% at IoU 0.75.
- Raw-bundle → identical CSV reproduction not complete; frozen GRU and cached target inputs reused.

## Evidence
Final run + explainability notebook: [`E06_wave5.ipynb`](E06_wave5.ipynb) (rebuilds the frozen CSV and checks its hash, then explains it).
[`wave5/PLAN.md`](PLAN.md), [`wave5/RESULTS.md`](RESULTS.md), [`results/wave5_*.json`](results/), [`docs/FINAL_METHOD.md`](../../docs/FINAL_METHOD.md), [`docs/final_method.json`](../../docs/final_method.json).
