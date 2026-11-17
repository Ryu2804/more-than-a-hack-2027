# Wave5: short-possession weighting passes validation, public score unchanged

> Report status: frozen campaign results underlying the team's final method. The cross-branch technical-report specification is in [FINAL_METHOD.md](../../docs/FINAL_METHOD.md) and [final_method.json](../../docs/final_method.json). The publication notes at the end record the situation when the campaign finished; this frozen candidate CSV was later published at `submission/artifacts/farras_wave5.csv`, without changing the results or the candidate hash. The submitted file (`carried_by_admin.csv`) is the same method retrained from scratch ([README](README.md)).

**Frozen candidate: `short_gru25/dead45`.** It passes the adoption rule written before training. External mean spellF1 rises under the same protocol; public spellF1 does not rise. The negative result on the paired ball-drop stress test is kept, not hidden behind the gain in short-spell recall.

## Decision before the public result

The primary objective remains provider-balanced mean spellF1, macroF1 as tie-breaker. A new candidate had to gain at least 0.01 over the control `old_gru25/wave4`, win at least 2/3 providers, and lose no more than 0.03 on any provider. Of 96 configurations, 6 qualify. The best eligible candidate is also the best raw alternative, so `selected` and `best_new` are the same role, not two independent models.

| Development provider | Wave4 control | Selected wave5 | spellF1 difference |
|---|---:|---:|---:|
| idsse | 0.583755 | 0.612109 | +0.028354 |
| skillcorner | 0.709044 | 0.709337 | +0.000293 |
| metrica | 0.676673 | 0.688852 | +0.012179 |
| Equal provider mean | 0.656490 | 0.670099 | +0.013609 |

SkillCorner development is almost flat; do not say every provider improved a lot. The worst development absolute count error drops from 44.62% to 41.88%, still above 30%. There is no claim that the short-spell/under-count problem is solved.

The selected candidate uses:

- 75% new PP-TCN with factor 2 on the state loss for live GT spells **< 4 seconds**, plus 25% frozen wave3 hybrid GRU.
- Short-spell length is measured on the full record within the half, not on the crop fragment. A run cut by unknown labels does not get a spurious short weight.
- Decoder `dead45`: dead threshold 0.45, min dead 1 s; H/A DP reset in each live interval; target rate calibrated on the **number of possessions after merging**.
- Distance gate 0.35 m, kappa 0.05, tau 3, merge 1 s, training rate scale 1. No dead-prior shift, no target GT and no target-specific kickoff rule.

Full rules hashed before fitting: [PLAN.md](PLAN.md).

## What was tried

The three new models keep the wave4 PP-TCN and its recipe: 1,500 steps/seed, batch 16, crop 384, AdamW lr 0.002 / weight_decay 0.01, OneCycle, dropout 0.15, seeds 0/1/2. The interventions are separated:

1. `boundary_tcn`: a third output head, auxiliary BCE weight 0.2, pos_weight 5; targets are known-state changes with a ±2-frame tolerance. The first crop frame and unknown adjacency are ignored.
2. `short_tcn`: double state loss on complete short live spells; sampler, two-head architecture and masking unchanged.
3. `masked_tcn`: full-ball-mask probability 0.6 instead of 0.3 in the base; noise and other parameters unchanged.

Eight decoders: the wave4 control, post-merge calibration, live reset, boundary-dependent transition cost, minimum dead 1 s, threshold 0.45, threshold 0.55, and no merge. The boundary-dependent DP uses the learned head when available; otherwise the change in conditional-team logit, not the 0.5 placeholder. The whole lambda grid is checked; the count is not assumed to be monotonic.

Twelve probability families cover the old PP/GRU control, the three new models and their 25% GRU mixtures, the three-model ensemble, and quality weighting from visibility and conditional entropy. No learned calibrator, provider-label switch or Metrica anchor tree.

| Family | spellF1 with wave4 decoder | Best decoder | Best spellF1 |
|---|---:|---|---:|
| old_pp | 0.649605 | boundary_reset | 0.652122 |
| old_gru25 | 0.656490 | boundary_reset | 0.658683 |
| boundary_tcn | 0.646890 | dead45 | 0.650364 |
| short_tcn | 0.659208 | boundary_reset | 0.661649 |
| masked_tcn | 0.646267 | postmerge | 0.648476 |
| boundary_gru25 | 0.655136 | boundary_reset | 0.660934 |
| short_gru25 | 0.664997 | dead45 | 0.670099 |
| masked_gru25 | 0.655079 | boundary_reset | 0.658034 |
| quality_old | 0.657831 | dead45 | 0.659872 |
| new_ensemble | 0.652994 | dead45 | 0.656644 |
| new_ensemble_gru25 | 0.660147 | boundary_reset | 0.664632 |
| quality_new | 0.659846 | dead45 | 0.666705 |

With the same decoder, the short-loss model beats the old PP. The auxiliary boundary head and full-ball-mask 0.6 do not beat the PP base on the matched control. Ensembling every model is not automatically better than the short model; adding weak models can dampen the gain. Decoder changes alone on the old GRU control do not reach the 0.01 adoption margin.

All 96 configurations and 2,208 development rows: [wave5_development.json](results/wave5_development.json). The configuration count is not the number of architectures or independent models. Cumulatively, five campaigns cover 298 configurations and 4,562 development evaluations, partly on the same matches.

## Data split and exposure

Training stays on the 19 licensed matches: 5 IDSSE and 14 SkillCorner. Six grouped folds exclude every round of the tested match; IDSSE uses 3 corruption rounds. Normalisation/rate come from the training fold only. Metrica 1/2 clean/degraded is zero-shot stress, not training for the new models or the GRU. The 23 development scenarios come from 21 matches; conditions within a match are averaged before the provider-balanced objective.

The cached PP development predictions are the original wave4 OOF. The final PP checkpoint is not used to fake held-out predictions on SkillCorner development. Shipped Ryu weights are not part of wave5, because they were trained on all IDSSE/SkillCorner matches. The inputs and code of every old wave stay read-only.

The old holdout covers 2 IDSSE, 6 SkillCorner and 1 Metrica match in two conditions. All of them were exposed in earlier campaigns. The 6 paired stress cases use the same SkillCorner matches with an extra transformation, not 6 new independent matches. Earlier public scores were also known; this is adaptive research, not a new blind test.

## Holdout after the selection freeze

| Cohort | Wave4 control | Selected wave5 | Difference in mean spellF1 |
|---|---:|---:|---:|
| IDSSE (2) | 0.644282 | 0.667175 | +0.022893 |
| SkillCorner (6) clean | 0.676204 | 0.678304 | +0.002100 |
| SkillCorner (6) stress | 0.664055 | 0.663030 | −0.001025 |
| Metrica 3, two conditions | 0.657178 | 0.690016 | +0.032838 |

On the 10 ordinary scenarios, selected wins 8/10; the two Metrica 3 conditions are not two independent matches. Averaged per unique match, selected wins 7/9. The Metrica 3 gain over wave4 does not exceed the historical wave3 mean of 0.701784. There is no claim of beating every campaign in every domain.

On the 6 paired stress cases, selected wins only 2/6. Mean spellF1 drops slightly; do not re-select the decoder or candidate based on this. All 64 role/scenario rows, including duplicate-role provenance: [wave5_holdout.json](results/wave5_holdout.json).

### Ball-loss transformation

After the freeze, seed 20261113 + a fixed match identifier, separately per half. An additional 35% of the originally visible ball rows are removed in contiguous spans of 10..60 samples. Labels, players and timeline are checked to be identical; all X16 and per-player/ball features are recomputed. It does not just blank the ball channel while leaking the old distance features.

| Match | Ball visible before | Ball visible after | Share of visible rows removed |
|---|---:|---:|---:|
| skillcorner_2010085 | 66.24% | 43.05% | 35.001% |
| skillcorner_2011166 | 54.38% | 35.35% | 35.000% |
| skillcorner_2013725 | 64.63% | 42.01% | 34.999% |
| skillcorner_2015213 | 63.50% | 41.27% | 35.002% |
| skillcorner_2016236 | 61.01% | 39.65% | 34.999% |
| skillcorner_2017461 | 56.46% | 36.70% | 35.000% |

Provenance summary without per-frame mask positions: [wave5_stress_summary.json](results/wave5_stress_summary.json). The stress GRU predictions were made on CPU in a separate process from the TCN, using the frozen wave3 checkpoint. Input/feature/prior SHA-256 were verified again before holdout/public.

## Short-spell diagnostics: they do not replace the main metric

All GT H/A spells remain in the main scorer. The table uses the official one-to-one same-team IoU matching at threshold 0.75 for GT **< 4 seconds**. An empty denominator is reported as null, not as perfect recall. The recall below is pooled from counts within the cohort for diagnosis; it is not the provider-balanced objective or a confidence interval. Metrica 3 uses the same GT in two conditions, so 254 is not 254 independent short spells.

| Cohort | GT short | Matched wave4 | Matched wave5 | Recall wave4 | Recall wave5 |
|---|---:|---:|---:|---:|---:|
| IDSSE (2) | 360 | 55 | 64 | 15.28% | 17.78% |
| SkillCorner (6) clean | 535 | 92 | 116 | 17.20% | 21.68% |
| SkillCorner (6) stress | 535 | 85 | 100 | 15.89% | 18.69% |
| Metrica 3, two conditions | 254 | 39 | 49 | 15.35% | 19.29% |

Short recall improves in every cohort, but there is a trade-off under stress:

| Stress, pooled, IoU 0.75 | Wave4 | Wave5 |
|---|---:|---:|
| Predicted possessions | 1613 | 1713 |
| Matched possessions | 744 | 751 |
| Precision | 46.13% | 43.84% |
| Recall of GT ≥ 15 s | 71.98% | 68.64% |

The pipeline produces more possessions, but part of the extra ones do not match GT; short recall alone does not guarantee a better total spellF1. This is a diagnosis of the full pipeline, not an isolated effect of the loss weighting, because the decoder changed too. All duration buckets/IoU and timing errors: [wave5_duration_diagnostics.json](results/wave5_duration_diagnostics.json).

## Public: no new spellF1 record

Four new fixed roles were evaluated after the freeze, holdout and independent refit. `best_new` is identical to `selected`, so there are only 3 distinct CSVs. Public GT 7,500 frames / 17 possessions; 12 predicted possessions in all three CSVs. There was no 96-way public sweep and no re-selection from public macroF1.

| Distinct CSV | SpellF1 | MacroF1 | Predicted possessions |
|---|---:|---:|---:|
| selected | 0.772414 | 0.902059 | 12 |
| previous_wave4 | 0.772414 | 0.895984 | 12 |
| pure_boundary | 0.772414 | 0.917189 | 12 |

`pure_boundary` has a higher public macroF1 than selected, but its external validation is lower and it did not pass the adoption rule. The candidate was not replaced after learning that result. The highest observed public score stays equal to wave4; the private/full-match score is unknown. [Public audit](results/wave5_public.json).

## Runtime verification and reproduction

- 10 regression tests pass: variable-cost DP vs exhaustive optimum, dead-interval reset, post-merge count calibration, boundary reset, unknown-label boundary supervision, full-spell short weights, visibility smoothing across halves, learned-boundary provenance, short GT retention, and unknown/half diagnostic splitting.
- Actual CUDA fitting/inference/checkpoint/refit of all three models passed with 4 steps/seed as a smoke test only. Saved probabilities and an independent smoke refit are identical; these are not mixed with the full-recipe results.
- The old control was re-run on the 23 development scenarios before full training and gave exactly the same objective, not merely a close one.
- Full training: 21 ensemble fits / **63 seed fits**, 94,500 optimiser steps, CUDA, exit 0. Recorded development/full-training time 5,037.110 s.
- After the selection freeze, the selected short-TCN component was actually retrained with 3 seeds / 1,500 steps on the same 19 matches; the frozen old GRU was verified by hash and is not claimed to be retrained in wave5.
- An independent rebuild (227.094 s) produced **199,319 target rows identical byte for byte**. The wave4 control CSV is also identical to its historical hash. The selection and all post-freeze input hashes stayed the same through the public audit.

Local candidate CSV: `out/candidates/selected.csv`.

CSV SHA-256:

`cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`

Frozen selection SHA-256:

`6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`

Additional aggregate evidence: [execution](results/wave5_execution.json), [verification](results/wave5_verification.json). How to rerun: [README.md](README.md). Source/PLAN/inputs were frozen before the real fits; later changes must become a new campaign.

## Publication and conclusion

Raw tracking, per-frame masks/labels/probabilities, model weights and the campaign's candidate CSVs are gitignored; only code and aggregate metrics/provenance are committed. The frozen candidate CSV is published in `submission/artifacts/farras_wave5.csv`; the submitted retrained file is `carried_by_admin.csv` at the repository root.

SkillCorner MIT, IDSSE CC BY 4.0. Credit and provenance for Ryu and the data are inherited from [E05_wave4/reference/SOURCE.md](../E05_wave4/reference/SOURCE.md). Wave5 adds no Metrica training and uses no anchor tree; Metrica remains event-proxy evaluation, not evidence of equivalence with the corrected target GT.

The short loss + new decoder are worth pursuing based on external validation, but they have not solved the long/short precision trade-off when the ball is lost. The next tests should protect long possessions and use fresh external validation when available; do not force gains on the same 17 public GT possessions. These results give no guarantee of private superiority.
