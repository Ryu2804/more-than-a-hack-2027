# Wave6: conservative fusion and occlusion-aware decoding

> English version of the pre-registered plan. The original plan, whose SHA-256 was recorded before the runs, was written in Indonesian; it is kept in git history at commit `efdb82e`, path `experiments/exp_farras/wave6/PLAN.md` (`git show efdb82e:experiments/exp_farras/wave6/PLAN.md`). Because this file was replaced, the recorded PLAN hash no longer matches.

## Basis for the decision

The audit of fetched refs covers main, farras, jeremy, rasya, ryu and consolidation. Commits and detailed evidence are in BRANCH_REVIEW.md. Consolidation compares different adapters and still uses wave2 for Farras, so it is not a ranking of the latest models. Rasya's branch has a transfer speed-unit bug and an orientation based on held-out GT; it is not imported. Ryu's branch has a nearest-opponent feature defect for opposite-view players; fixing it needs retraining and must not be applied unilaterally to the old checkpoint. Jeremy's exp7 fixes the simulated damage but fails the adoption intersection. A recovered-ball detector that failed the independent gate is not merged.

Wave5's short loss gives external mean spellF1 0.670099305497763, but the paired missing-ball stress test adds false possessions and lowers long-spell recall. Wave6 hypothesis: conservative mixtures, separating the dead/team heads, and visibility-aware decoding can hold back false transitions without dropping short GT. No change to learned features, no new training and no change to the label definition.

## Fixed grid: 36 alternatives + one control

All bundles come from frozen probabilities: short_gru25, old_gru25, boundary_gru25; each is already 75% TCN + 25% GRU. Cached development TCN predictions must be OOF, not final-model predictions on a training match. No target GT, provider-dependent switch or short-GT cutoff.

1. m01_blends: joint mixture of short with old or boundary, other weight 0.10/0.25/0.50.
2. m02_visibility: short weight = missing + (visible − missing) × mean11(visibility), visible 0.85/1.0 and missing 0.25/0.50/0.65; the rest is old. The centred window is cut at the half.
3. m03_heads: the short conditional-team probability is kept; the short dead probability is mixed with old/boundary with other weight 0.25/0.50/0.75. Joint probabilities are reconstructed.
4. m04_occlusion_cost: short kept, switch-penalty multiplier 1 + (factor − 1) × (1 − mean_window(visibility)), factor 1.25/1.5/2.0, window 1/11. The cost applies to the transition into a frame and does not cross a dead/half reset.
5. m05_dead_hysteresis: connected runs with p_dead > low, low 0.35/0.40, high 0.50/0.55/0.60. A run needs at least 5 samples and must contain p_dead > high; separately per half.
6. m06_short_guard: only a low-confidence occluded A-B-A excursion may be merged. B lasts ≤ 1/2/3 seconds, mean ball-visible < 0.5, mean conditional confidence of B < 0.60/0.70; both A neighbours must be the same, live and in the same half. Decisions are simultaneous over the original runs, not cascading. Strong short possessions are kept.

Control: the exact frozen wave5 short_gru25/dead45. Default decoder dead threshold 0.45 / min 1 s, H/A live-reset DP, gate 0.35 m, kappa 0.05, tau 3, merge 1 s. All 18 lambdas are checked; the target count comes from the training fold; calibration uses the number of possessions after merging AND the custom post-processing. Count is not assumed monotonic. No prior shift, target-specific kickoff, rate tuning or 36-way public sweep.

## Data and objective

The six grouped folds are reused: 19 training matches (5 IDSSE, 14 SkillCorner), with the folds' original normalisation/probabilities/rate. Development 23 scenarios / 21 unique matches; Metrica 1/2 clean + r0 is zero-shot evaluation only. Conditions within a match are averaged first, then the mean of matches per provider, then the equal mean of the three providers. The primary is the official spellF1 over all H/A spells; macroF1 only as tie-breaker.

Rank by spellF1, macroF1, identifier. An alternative is eligible if its gain over wave5 is ≥ 0.005, it wins at least 2/3 providers, no provider drops > 0.01, and the lower paired match-bootstrap 95% bound is > 0. Bootstrap 5,000, seed 20261114, resampling unique matches WITHIN provider, with all conditions of a match moving together. The percentile interval is not corrected for multiple comparisons and uses adaptive development data; it is not confirmatory evidence or a probability of a private win. If nothing is eligible, the control stays selected. The best raw alternative is still reported, not removed.

## Exposure and evaluation stages

All old data, previous public scores and the wave5 paired stress test have been exposed. This is adaptive exploration, not a fresh blind holdout. No private/full-match performance claim. The 6 stress cases come from the same 6 SkillCorner holdout matches, not 6 additional matches. Ordinary holdout 10 scenarios / 9 unique matches; with stress, 16 scenarios, still 9 unique matches.

Source/PLAN, cached probabilities, old manifests, data and external-source digests are frozen before the new evaluation. The selection is frozen before the wave6 holdout/public. Three fixed roles: selected, best_new, previous_wave5; duplicate roles are not independent models. No re-selection based on holdout, stress or public macroF1.

An independent-process rebuild reloads the frozen probability caches and repeats decoding until the target CSV is identical (199,319 rows). This is not a model refit; refit evidence for the learned components is inherited from wave5/wave3. An inference pipeline that needs a missing cache must fail, not silently write or regenerate old campaign results.

## Execution and publication limits

Six implementation slices; the parent owns the protocol/runner/selection. The actual evaluation runs in one CPU process because about 1.57 GB of RAM is available; six Torch processes are not run. No new training seed fits. All methods use the same source API and do not copy the whole pipeline.

Regression checks only for real edge cases: normalisation/conditional heads, half-local visibility, hysteresis seed/min-duration/half boundary, transition-cost semantics, simultaneous A-B-A corrections, preservation of high-confidence short spells, post-processing count calibration. Run them once after integration, before the freeze. If one fails, fix it before the freeze and rerun the failing check.

Raw tracking, per-frame labels/probabilities, weights and submission CSVs stay ignored in cache/out. Code, the branch audit and aggregate results may be shared. No automatic commit/push; teammates' source and wave1..5 are not modified.
