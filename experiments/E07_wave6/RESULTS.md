# Wave6: hysteresis improves, but does not pass the promotion rule

The audit covers six remote branches. Six paths produce 36 alternatives + 1 control, 851 development evaluations on 23 scenarios / 21 unique matches. No new model training. Every result comes from decoding/fusion of frozen probabilities. [Branch audit and list of fixes](BRANCH_REVIEW.md), [plan before evaluation](PLAN.md), [how to run](README.md).

## Frozen decision

The best alternative is **`m05_dead_hysteresis/low35_high50`**. Mean development spellF1 **0.674807**, versus wave5 **0.670099**, gain **0.004707**. The rule set before the run required a gain of at least 0.005, so it is **not promoted**. It misses the threshold by about 0.000293; the threshold was not lowered after seeing the result. The selection stays wave5 `short_gru25/dead45`.

The paired match-bootstrap 95% interval of the best alternative's gain is [0.003412, 0.006096], 5,000 draws, seed 20261114. This is an exploratory development interval, not corrected for the 36-way search or for adaptation across campaigns. A positive lower bound does not override the 0.005 minimum-effect requirement. There is no claim of statistical confirmation on unseen matches.

| Development provider | Wave5 control | Best alternative | spellF1 difference |
|---|---:|---:|---:|
| IDSSE (5) | 0.612109 | 0.609661 | −0.002448 |
| SkillCorner (14) | 0.709337 | 0.718859 | +0.009522 |
| Metrica 1/2 | 0.688852 | 0.695901 | +0.007049 |
| Equal provider mean | 0.670099 | 0.674807 | +0.004707 |

Two providers win and IDSSE drops slightly. The best alternative's worst development absolute count error is 42.66%, versus about 41.88% for wave5; the count problem is not solved. Development macroF1: best alternative 0.885634, control 0.885922. The choice was not reversed based on public macroF1.

## What the best alternative changes

The model and joint probabilities stay wave5: 75% short-loss PP-TCN + 25% frozen GRU. No retraining, no change to player features, no target labels, no touch detector or recovered-ball predictions.

One change: the dead gate. In each half, take each connected run with p_dead > 0.35 of at least 5 samples at 5 fps, and accept the whole run only if it contains a sample with p_dead > 0.50. Weak runs without a strong seed are rejected. Samples before the strong seed can also become dead, so the method stays offline. Original wave5 accepts only contiguous p_dead > 0.45 lasting at least 1 second.

The H/A Viterbi is still reset per live interval; distance gate 0.35 m, kappa 0.05, tau 3, merge 1 s. All 18 lambdas are checked with the target rate from the training fold; the count is taken after merging. No dead-share shift and no target-specific kickoff rule.

## Results of the six paths

| Path | Best development configuration | SpellF1 | Difference vs control |
|---|---|---:|---:|
| Joint blends | boundary10 | 0.669565 | −0.000534 |
| Visibility-adaptive blend | visible100_missing65 | 0.669300 | −0.000799 |
| Dead-head-only blend | boundary75 | 0.670813 | +0.000713 |
| Occlusion switch costs | factor1.25_window1 | 0.668912 | −0.001188 |
| Dead hysteresis | low35_high50 | 0.674807 | +0.004707 |
| Low-confidence short guard | max1_confidence60 | 0.670099 | 0.000000 |

No configuration is eligible. Six hysteresis settings take the top six ranks, but every gain is still < 0.005. More conservative blending, visibility gating and larger switch penalties do not automatically help. The best short guard scores the same as the control, not an improvement.

All 37 configuration records and 851 per-scenario scores: [wave6_development.json](results/wave6_development.json). The 36 configurations are not 36 architectures or training runs. Development runtime 82.344 s in the script, one CPU process, training_seed_fits 0: [execution](results/wave6_execution.json).

## Holdout after the freeze

Three fixed roles: selected, best_new, previous_wave5. Selected and previous_wave5 are identical, so there are only two distinct pipelines. 48 role/scenario rows cover 16 scenarios from 9 unique matches; the 6 stress cases come from the same 6 SkillCorner matches. Every match and the stress transformation were exposed in older campaigns.

| Cohort | Wave5 control | Best hysteresis | Difference in mean spellF1 |
|---|---:|---:|---:|
| IDSSE (2) | 0.667175 | 0.663522 | −0.003652 |
| SkillCorner (6) clean | 0.678304 | 0.684029 | +0.005726 |
| SkillCorner (6) stress | 0.663030 | 0.667709 | +0.004679 |
| Metrica 3, two conditions | 0.690016 | 0.698456 | +0.008440 |

The best alternative wins 7/10 ordinary scenarios, or 6/9 unique-match means. Mean gain on ordinary scenarios 0.004393. On the paired ball-drop stress test it wins 6/6, mean gain 0.004679. This improves the stress result over wave5, but it is not a reason to re-select a candidate that failed the development requirement. Metrica 3 clean/r0 use the GT of the same match, not two independent matches.

### Precision and long possessions under stress

Diagnostic with the official one-to-one matching at IoU 0.75, pooled over the six paired stress cases:

| Diagnostic | Wave5 | Best hysteresis |
|---|---:|---:|
| Predicted possessions | 1713 | 1669 |
| Matched possessions | 751 | 760 |
| Precision | 43.84% | 45.54% |
| Recall GT < 4 s | 18.69% | 20.00% |
| Recall GT ≥ 15 s | 68.64% | 68.12% |

The predicted count drops by 44 and the matched count rises by 9; precision and short recall improve. Long-spell recall still drops slightly, so there is no claim that the whole robustness trade-off is solved. All 1,689 GT spells remain in the main scorer. This IoU-0.75 recall is a diagnostic, not a replacement for the headline mean over IoU 0.3..0.7.

[Full holdout](results/wave6_holdout.json), [comparison aggregates](results/wave6_comparison.json), [duration diagnostics](results/wave6_duration_diagnostics.json).

## Public diagnostic: a new score, not a promoted candidate

The public GT is only 7,500 frames / 17 possessions. Only the three fixed roles were evaluated, after the selection, holdout and selected rebuild; selected/previous are identical. No 36-way public sweep.

| Distinct pipeline | Public spellF1 | Public macroF1 | Predicted possessions |
|---|---:|---:|---:|
| Selected wave5 | 0.772414 | 0.902059 | 12 |
| Best hysteresis, exploratory | **0.786207** | **0.925369** | 12 |

The best alternative gives the highest public spellF1 diagnostic recorded in our campaigns, +0.013793. **It is still not promoted**: the development minimum effect failed before the public window was opened. A public gain does not prove a private improvement and is not used to lower the adoption threshold. The private/full-match score is unknown.

Local best-alternative CSV: `E07_wave6/out/candidates/best_new.csv`.

SHA-256: `2bf7cc52306740e4af5ca1e34c07d843e3d7705b0f166ad051d414bd5f4c5acf`.

Selected stays `E07_wave6/out/candidates/selected.csv`, identical to the wave5 candidate.

SHA-256: `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.

[Public audit](results/wave6_public.json).

## Verification actually run

- 40 regression cases: 39 passed in the first batch; one failed because `import run` resolved to the wave5 runner instead of wave6. The test import was changed to a unique module identity, then only the failing case was rerun and passed. The assertions are unchanged. All fixes were finished before pre-registration and the real evaluation.
- The checks cover joint-vs-conditional mixtures, zero-live heads, half-local visibility, actual variable-cost decoder parity, hysteresis strict thresholds/min duration/half reset, the simultaneous A-B-A guard and strong-short preservation, count calibration after custom post-processing, and bootstrap clustering of conditions within a match.
- The actual dev CLI ran 37 settings on 23 scenarios and reproduced the control objective 0.670099305497763 exactly. Holdout ran 48 role/scenario evaluations; the official public scorer accepted both distinct CSVs.
- A separate process reloaded the cached probabilities and repeated the selected decoding. The 199,319 target rows have the same SHA as wave5. This is a selected **decoding rebuild, not a model refit**. The best alternative has no new independent model refit; no new fits were made for any pipeline.
- Frozen source/input/selection hashes were checked at the holdout, rebuild and public stages. Selection SHA `e387c08beef727217ffeb8385eeca722c1348a554d38e2b087cd1b4c2fa099ec`.

[Rebuild evidence](results/wave6_verification.json) and [regression/final check notes](results/wave6_checks.json). The temporary rebuild CSV was deleted after the hash comparison. No source/PLAN/input edits after pre-registration.

## Repository fixes applied and not applied

Wave6 adds a fail-closed cache check before inherited inference, exact conditional-head preservation when the live mass is 0, bootstrap per unique match, a common decoder API for the six paths, hash checks, and an explicit distinction between selected/exploratory/public/private. No fallback writes results of old campaigns.

The audit found that the consolidation registry wrongly labelled some of Jeremy's **dead shares** as public spellF1; Farras's benchmark there is still the wave2 adapter; Rasya's branch has a speed-unit bug and a held-out GT orientation; Ryu's branch has a nearest-opponent feature defect that our snapshot also inherits. Details, sources and risks are in [BRANCH_REVIEW.md](BRANCH_REVIEW.md). These findings are **not** claimed to be fixed on teammates' branches. Fixing a learned feature needs regeneration + retraining, not an inference-only patch on old weights.

Teammates' source and wave1..5 are intact. Raw tracking, per-frame probabilities/labels, weights and submission CSVs stay ignored.
