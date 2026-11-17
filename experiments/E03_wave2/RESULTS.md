# Farras experiment results (wave1/wave2 history and campaign index)

## Team final method

The team uses **wave5 `short_gru25/dead45`** as the final method. The technical report follows the [cross-branch specification](../../docs/FINAL_METHOD.md) and the [Typst facts](../../docs/final_method.json). This decision uses the frozen selection that passed the external adoption rule, not a public evaluation or new training. [Candidate identity/hash](../../submission/method.json), [export and readiness status](../../submission/README.md), [wave5 results](../E06_wave5/RESULTS.md).

Wave6 and the later campaigns remain comparison research. Primary/control decisions in each benchmark apply only to its own protocol. The history below is not rewritten. The package is not ready for official submission because the raw bundle → identical CSV reproduction is not complete.

For the team harness, one selected prediction CSV is published at `submission/artifacts/farras_wave5.csv`. Its hash is identical to the frozen wave5 candidate. Publishing the campaign source and evidence did not retrain models or run new scoring; caches, weights, raw data and GT are not included.

## Later research

Latest generalisation work: [E13_provider_generalization/RESULTS.md](../E13_provider_generalization/RESULTS.md). Two-way source-provider-only training without competition GT, 30 seed fits, five recipes. Control equal-provider spellF1 **0.540864**; RDrop **0.540882**, a practically zero difference with an interval crossing 0. The primary, modality dropout, at **0.530378** failed promotion. The control was kept. 58 tests passed / 1 skipped, 300 artifacts sealed before scoring, and 270 probability/decoded arrays and metric rows reproduced identically. **This does not guarantee being overfit-free:** all 27 external matches had been exposed historically; there is no fresh blind cohort yet. Not directly comparable with the older IDSSE ensemble numbers. [Exposure audit and read-isolation limits](../E13_provider_generalization/EXPOSURE.md).

Six further recipes: [E12_event_learning/RESULTS.md](../E12_event_learning/RESULTS.md). The best exploratory event+contact ensemble reached mean IDSSE spellF1 **0.624577**, up 0.005014 from the strong research control 0.619563, winning 5/7; the interval still crosses 0. The primary, event+balanced at 0.619192, failed promotion and the strong control was kept. Ranking/dead45 reached 0.621512 and won 7/7 against the strong consensus control; the same-decoder contrast is not clearly separated. 126 fresh seed fits, 77 tests, 42 saved-model replays and 280 decoded arrays identical. No new public/private score.

Latest new training: [E11_spell_training/RESULTS.md](../E11_spell_training/RESULTS.md). The best exploratory event ensemble + consensus reached IDSSE spellF1 **0.619563**. The primary, event ensemble/dead45, reached **0.618941**, up 0.003484 from control 0.615458, winning 5/7 with a positive interval but below the 0.005 minimum gain. The control was kept. 63 new seed fits, 27 tests passed, 21 saved-model inference replays and 140 decoded arrays identical. The standalone event TCN also improved, 0.608023 versus the old short-TCN 0.595755; no new public/private score.

Latest new methods: [E10_temporal_methods/RESULTS.md](../E10_temporal_methods/RESULTS.md). The best exploratory consensus reached IDSSE spellF1 **0.616970**, up 0.001512 from control 0.615458, but the interval of the difference still crosses 0. The primary HSMM at 0.605838 failed promotion; the control was kept. 19 settings / 133 evaluations, 47 tests passed, 133 decoded arrays and 7 refitted priors identical. No new public/private score.

Latest cross-branch adaptation: [E09_cross_branch/RESULTS.md](../E09_cross_branch/RESULTS.md). The combined contact/memory/PP-TCN/short-loss/SetSeq primary reached IDSSE spellF1 **0.609030** against control 0.595755, winning 6/7 and passing the research rule. The best raw alternative, short75+SetSeq25, reached **0.615458**, winning 7/7; it does not replace the primary post hoc. All 28 settings and the negative results are available; no new public/private score.

Latest shared benchmark: [E08_fair_idsse/RESULTS.md](../E08_fair_idsse/RESULTS.md). Mean IDSSE spellF1: short-loss PP-TCN 0.595755, SetSeq 0.578602, PP-TCN 0.571090, HGB 0.315719. All learned arms were retrained on the other 6 matches in each fold and used the same decoder. Not an exact submission ranking or a public score; the short-TCN versus SetSeq interval still crosses 0.

The latest public diagnostic remains [wave6](../E07_wave6/RESULTS.md): 0.786207 for the best exploratory alternative; the selection stays wave5 (public 0.772414). Nothing was re-selected from the public GT. The earlier feature audit was corrected in the benchmark report: Ryu's reference and our actual builder are not the same function. History: [wave5](../E06_wave5/RESULTS.md), [Ryu/wave4](../E05_wave4/RESULTS.md), [Jeremy/Farras/wave3](../E04_wave3/RESULTS.md). The report below keeps the decisions and historical numbers of the first two batches.

## Wave1 and wave2 history

72 configurations in two batches, 288 external development evaluations. The main metric is spellF1; frame macroF1 is only a selection tie-breaker. All public scores below are local evaluations on the same public GT window, not private leaderboard results.

| Comparison | Public spellF1 | Note |
|---|---:|---|
| Official baseline smooth50 | 0.325581 | Public reference |
| Ryu result reviewed before the campaign | 0.528571 | Control gate; not a claim about the latest Ryu branch score |
| Wave1, externally selected candidate | 0.442105 | Hysteresis radius 1.5 m |
| Wave2, externally selected candidate | 0.460606 | H/A augmentation + histogram boosting + Viterbi 6 |
| Wave2, highest public diagnostic | 0.560000 | 7 control features + histogram boosting + Viterbi 6; not the external choice |

The full/private baseline published by the competition is 0.2433. It is not directly compared with the public numbers in the table; the private score of these experiments is unknown.

## The configuration with public 0.56

`E03_wave2/methods/w14_feature_ablation.py`, configuration `control`:

1. Uses the `atd_full.npy` and `team_dist.npy` caches from Ryu's pipeline for the target. Does not modify the original pipeline.
2. Uses 7 features: each team's nearest-player distance to the ball, the margin between the two distances, ball visibility, ball distance to the pitch boundary, gap age and gap remaining. Absolute position, speed and player-shape features are not used.
3. `HistGradientBoostingClassifier`: max_leaf_nodes 15, max_iter 100, learning_rate 0.08, l2_regularization 1, early_stopping False, seed 2027. Balanced class weights, at most 40,000 labelled training rows.
4. Trained on Metrica Games 1/2, original condition plus a synthetically occluded copy. Target GT is not used for training.
5. Viterbi switch penalty 6, separately per half. No team-swap augmentation in this configuration.
6. Predictions at 5 fps are expanded to 25 fps; gap remaining makes the pipeline offline, not real-time.

External development mean spellF1: 0.500969, macroF1: 0.728420. Public diagnostic spellF1: 0.560000, macroF1: 0.650520, 13 predicted possessions versus 17 GT in the public window.

Public 0.56 was not used to replace the frozen candidate. Declaring it the winner after seeing the public GT would mix the test diagnostic with model selection.

## Frozen candidates and holdout

| spellF1 evaluation | Wave1 | Wave2 |
|---|---:|---:|
| Mean Games 1/2, clean/occluded | 0.460517 | 0.517438 |
| Game 3 original condition | 0.474935 | 0.599664 |
| Game 3 added occlusion | 0.374534 | 0.433982 |

Wave2 frozen: `w08_symmetric_boost/cost6`. H/A features are swapped together with the margin and labels in training, followed by histogram gradient boosting + Viterbi 6. Target CSV SHA-256:

`45e03662fccbeb238e9df723c99829df60e3d8d641d539be9f6d1e1fd59b0446`

Wave1 candidate SHA-256:

`6a022089301a08d9d469fbc4630cb65c3f4a34b1d4de1193488422370e63b52d`

Game 3 was not used for training/tuning or picking the winner. Only the two frozen candidates were tested on both conditions. The parser checks 11 metadata channel layouts, 35 player identities and the event periods to avoid player-slot and half errors.

## Negative findings and limits

Shape-only features are weaker than the control features under this protocol. GaussianNB, a confirmation gate, a learned transition logistic and dead-ball bridging did not beat the selected candidate. More complex models do not automatically give a higher spellF1.

All 72 configurations use the same two development matches; the search is adaptive. Game 3 is a single holdout match, and its labels are event proxies that are not identical to the challenge definition. Added occlusion still clearly lowers spellF1.

The Ryu score above comes from reviewing an earlier experiment cache. The team's remote branches evolved after that review; there is no claim that 0.56 beats every member's latest result. There is no equivalent private/full-match comparison.

## Pushed evidence

`results/` stores aggregate scores for all configurations, candidate configurations, holdout metrics and reproduction hashes. The local `out/` stores per-frame CSVs/predictions and the pre-registration; it is gitignored. Raw data and caches are gitignored as well.

After moving to this repository, both campaigns were rerun with the same inputs. The move and import changes did not alter the algorithm or the configuration. The reproduction result and the comparison with the original campaign are recorded in `results/relocation_verification.json`.

Commands and input requirements: [README.md](README.md) and [CAMPAIGNS.md](../CAMPAIGNS.md).
