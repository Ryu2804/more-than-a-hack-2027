# Shared IDSSE benchmark: short-loss PP-TCN highest on the mean

The run completed all 7 folds, 63 neural seed fits, 31,500 optimiser updates and 7 HGB fits. No partial scores or old checkpoints are mixed in. A separate process reproduced all 63 decoded arrays identically. [Protocol before the run](PLAN.md), [reproduction](README.md), [all results](results/fair_idsse_results.json).

## Main result: same decoder

Every learned arm received the same prepared r0 inputs and provider GT, 7 leave-one-match-out folds, each training on exactly the other 6 matches. Neural arms use seeds 0/1/2, 500 updates/seed, batch 16, window 384. Every probability uses the wave5 dead45 decoder, without per-model or per-outer-fold tuning. No target/public GT.

| Adapted family | Mean spellF1 | Mean macroF1 | Mean count error |
|---|---:|---:|---:|
| Short-loss PP-TCN, Farras-style | **0.595755** | **0.902935** | −23.98% |
| SetSeq, Jeremy-style | 0.578602 | 0.896519 | −32.51% |
| PP-TCN, Ryu-style | 0.571090 | 0.900344 | −33.82% |
| HGB symmetric, Farras wave2-style | 0.315719 | 0.771451 | +1.13% |

A count close to the GT does not guarantee correct possessions. HGB has a small mean count error but a spellF1 far below the neural models. So HGB's low result is not just a problem of the old table: on the new benchmark this older model family still trails. Our newer temporal pipeline is a different comparison from the old HGB.

**This is not an exact ranking of each branch's submission.** SetSeq is the wave3 adapter, not Jeremy's exp7; the PP-TCN is retrained on 6 IDSSE matches only, not Ryu's 27-match checkpoint; the short-loss PP-TCN does not use the frozen wave5/6 GRU mixture. Do not relabel this table as a final ranking of team members.

### Paired uncertainty

- Short-loss PP-TCN versus PP-TCN: gain 0.024666, exploratory paired-bootstrap 95% [0.018060, 0.032109], wins 7/7 matches.
- Short-loss PP-TCN versus SetSeq: gain 0.017153, interval [−0.003553, 0.038950], wins 5/7 matches. The interval crosses 0, so there is no strong evidence that the short loss beats SetSeq on unseen matches.
- SetSeq versus PP-TCN: gain 0.007513, interval [−0.018959, 0.030560], wins 5/7 matches; not clearly separated.

Bootstrap 5,000 draws, seed 20261114, unit = unique match, not frame or seed. All 10 pairs are in the JSON; the intervals are exploratory and not corrected for multiple comparisons. The seven matches had been exposed in earlier research; this is not a fresh blind test.

## Separate rule baseline

The adapted touch-hold gives spellF1 **0.200755**, macroF1 **0.600833**, mean count error +31.74%. This is a native hard-label rule panel, not the shared probabilistic decoder. The rule uses the same corrupted 5-fps tracking, team metadata without a GT flip, speed from position in canonical metric coordinates, and no oracle dead/X mask. This number is **not a reproduction** of Rasya's 0.5205 in the old table.

This does not mean all of Rasya's code/submission can only reach 0.2008. The 5-fps adaptation, fixed cues, missingness, and the removal of the GT-dependent orientation can change the result. No ablation here isolates the contribution of each change.

## Argmax diagnostic, not a re-selection

| Family | Argmax spellF1 | Argmax macroF1 |
|---|---:|---:|
| HGB | 0.041246 | 0.717949 |
| SetSeq | 0.567694 | 0.898332 |
| PP-TCN | 0.560026 | 0.900524 |
| Short-loss PP-TCN | 0.571138 | 0.901641 |

All use the same argmax over the 3-class probabilities; the result shows why the temporal decoder matters, especially for HGB. This result is not used to pick a model-specific decoder. The main table keeps the decoder fixed in the PLAN.

## What is now more equal

1. Every learned fit gets the same 6-match pool and excludes the test match entirely. No truncation to the first 2 matches as in the old consolidation HGB adapter.
2. Ryu's shipped checkpoint, which had seen the seven benchmark matches, is not loaded. All 21 neural ensemble fits start from scratch.
3. Predictors receive only observations; y/y25 are removed. The neural API rejects training/test overlap and label keys. Home/away metadata is not flipped based on held-out GT. The rule does not copy the GT dead/X mask.
4. The same original provider GT, all short spells in the official scorer, the same timeline/half mapping for every output.
5. Model normalisation and the decoder's rate/dead_share are computed from the training fold only. The learned arms' decoder is identical; the native rule is clearly separated.
6. Neural updates/seeds/batch/window are the same; HGB's model-specific budget and every optimiser/loss/sampling choice are reported, not disguised as a causal architecture-only ablation.
7. Source/PLAN/data/features were frozen before the full fits. New caches bind the manifest, training membership, held-out match, family, seeds, steps and prediction SHA; a mismatch fails.

The benchmark does not guarantee that every design choice is optimal for each architecture. Even the common decoder may suit some families better. The 500-update budget is lower than the 1,500-step production recipe; the results are not each model's ceiling. Do not compare the main-table numbers directly with wave6's public spellF1 or with the three-domain provider-balanced objective of wave5/6.

## Runtime and verification

- 32 semantic tests pass before the fits: 6-match exclusion, GT stripping, half-local rate, actual HGB pooling/fit, rule visibility/metres-per-second/reset/metadata, neural train/test isolation and averaging of joint probabilities.
- The combined smoke test actually finished computing but Windows exited fatally (−1073740791). The SetSeq-only CUDA child also exited with 3221226505. Both runs are recorded as failures, not accepted because a JSON output existed.
- Backend before the full freeze: SetSeq CPU; PP/short-TCN CUDA; HGB/rule CPU. The corrected isolated family smoke tests exited 0. Equal updates are not equal hardware/FLOPs; no latency/efficiency ranking.
- Full benchmark plus independent rebuild completed with exit 0. Full script time 15,565.016 s; most of it is SetSeq CPU training (aggregate fit 13,791.594 s). Every model/fold runs separately, so a child's exit 0 is verified before the parent reads the cache.
- 35 job-membership records checked: learned actual_fit_names are exactly the other 6 names; rule actual_fit_names are empty. 63 score rows cover 35 primary rows and 28 argmax diagnostics. The 63 full neural seed fits do not include the reduced smoke fits, whether the smoke failed at runtime or was accepted.
- An independent-process decode rebuild reloaded 35 prediction caches and reproduced 63 identical label-array hashes. This is not an independent model refit and not a 199,319-row target rebuild, because the target is not evaluated.

Manifest SHA: `9e47a83be188a497aa2c0b6f4e02f0f231235d94c65dcd61b2f816ecd378dff9`.

[Protocol hashes](results/fair_idsse_protocol.json), [runtime/isolation checks](results/fair_idsse_checks.json), [decode rebuild](results/fair_idsse_verification.json).

## Correction of the earlier feature audit

The earlier claim that our model automatically inherits the nearest-opponent bug of Ryu's reference copy is wrong. The reference file does use a view-relative mask, but the actual builder `wave4/features.py:_player_set` uses per-player team inequality and rejects more than 48 present players. The benchmark uses the actual builder's feature packs, not that reference function. No wave4/5/6 algorithm is changed in this run; what is corrected is the audit's explanation, not the old frozen results.

## Publication

New code only in `fair_idsse/` (now `E08_fair_idsse/`); raw tracking, per-frame probabilities/labels, weights and caches stay ignored. Only 4 aggregate JSONs are exported to `results/`. The new models are not committed and are not submission candidates. The private/full-match score is unknown. Teammates' source and old campaigns are not modified; the Farras index records the new benchmark and its comparison limits.
