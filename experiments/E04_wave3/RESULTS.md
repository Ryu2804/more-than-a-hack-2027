# Wave3 results: combining Jeremy's and Farras's approaches

## Decision

66 configurations completed: 18 neural fits, 18 IDSSE tree fits, 594 development evaluations, 16 holdout candidate/scenario evaluations and 4 public sanity checks. Across Farras's three batches: 138 configurations and 882 development evaluations. These are not 138 independent models; several decoders/mixtures share the same model probabilities.

Candidate frozen before holdout/public: **hybrid_tree25 + centered2**. State probabilities are 75% from the identity-free set+BiGRU model with Farras's 7 control features added, and 25% from the Metrica-trained control tree. Boundary probabilities are mixed with the same weights; the tree anchor uses a neutral boundary of 0.5. Decoder: lambda_team 2, lambda_dead 2, gamma 0.5, temperature 1.

Mean development spellF1 **0.615394**, macroF1 **0.873323**. This number is the best of a 66-configuration development ranking, not a selection-bias-free test estimate. It is not directly comparable with Jeremy's 0.656 proxy, because dataset split, model, token sampling, recipe, decoder and objective differ.

## Public: better than the old Farras candidate, not better than Jeremy's original

| Pipeline | Public spellF1 | Public macroF1 | Predicted possessions |
|---|---:|---:|---:|
| Jeremy original, branch report | 0.581818 | 0.725033 | 16 |
| Frozen wave3 combination | 0.575610 | 0.712844 | 24 |
| Old Farras control | 0.560000 | 0.650520 | 13 |
| Local SetSeq adaptation + cost 0.5 | 0.526316 | 0.739575 | 21 |
| Farras wave2 frozen | 0.460606 | 0.711830 | 16 |

The window has 17 GT possessions. The new candidate is 0.015610 above Farras's 0.56 diagnostic but 0.006208 below Jeremy's reported 0.581818. Neither difference is shown to be significant; Jeremy's interval is wide and the window is small. The new candidate's macroF1 is also below the SetSeq adaptation, although its spellF1 is higher. 24 predicted possessions indicate over-segmentation on this window.

Only four pre-specified candidates were scored on public after the freeze. There was no 66-way public sweep, no parameter change and no replacement of the winner after seeing the score. The public result is not a private/full-match score and is not a reason to adopt this candidate in place of Jeremy's submission.

## Holdout: comparison on the same inputs

The two IDSSE holdout matches were not used for training or configuration selection in this campaign, but both were used in Jeremy's research. Metrica Game 3 had been evaluated in wave2. All these limits were known before training.

| spellF1 scenario | Combination | SetSeq adaptation | Old control | Old symmetric |
|---|---:|---:|---:|---:|
| IDSSE J03WQQ degraded | 0.657420 | 0.647263 | 0.249697 | 0.283692 |
| IDSSE J03WR9 degraded | 0.537888 | 0.501046 | 0.254669 | 0.294098 |
| Metrica 3 original | 0.740230 | 0.723864 | 0.589509 | 0.599664 |
| Metrica 3 degraded | 0.663339 | 0.639759 | 0.359310 | 0.389950 |

The combination beats all three references in all four scenarios. Mean of the two IDSSE matches: 0.597654 versus 0.574155 for the SetSeq adaptation. Mean of three matches, averaging Metrica 3 clean/degraded first: 0.632364 versus 0.610040. Only three matches, and no paired CI suitable for a population claim.

Metrica 3 degraded uses Jeremy's degrader with player gaps, permutations, duplicate slots and ball gaps, not wave2's ball-only occlusion. Because the conditions differ, do not compare 0.663339 directly with the old wave2 occluded score of 0.433982.

## What the ablations show

All deltas below are development deltas, not independent holdout ablations. The grid was not changed after seeing results.

- SetSeq + cost 0.5: 0.607472. Hybrid with the 7 control features on the same decoder: 0.609823. Adding the control features gives a small delta of 0.002351, not a jump proven to be stable.
- On the same centered2 decoder, hybrid 0.613449 and hybrid_tree25 0.615394: adding the 25% tree gives 0.001946. The mixture's contribution is small on development, although the full candidate is better on holdout.
- On the same hybrid_tree25, constant cost 2 gives 0.603684 and centered2 gives 0.615394: delta 0.011710. This supports trying the centered boundary in a new campaign; it does not overturn Jeremy's report that his original follow-up was inconclusive.
- Best feature-only GRU 0.398470, far below the best SetSeq 0.610136. Both were trained on the same folds, but the set model also uses mirroring; this is not a pure architecture isolation.
- The Metrica-only tree anchor's best is 0.306517 on the combined objective, even though it is still 0.56 on public. The difference in training domain means this is not a fully equal comparison of architecture capacity.
- Raising the tree weight to 75% lowers the best score to 0.480214. The visibility gate (0.534255) also does not beat the fixed 25% weight. The hypothesis that the tree should dominate while the ball is visible is not supported under this protocol.
- The selected candidate still under-segments on all 5 IDSSE development matches: worst count error −41.68%, failing the 30% quality bound recorded as a warning. On the two IDSSE holdout matches the error is −20.20% and −25.95%, but the public possession count is 41.18% too high. One global count prior is not yet safe across providers.

## Full ranking of the 66 configurations

The score is the mean over 7 matches: 5 IDSSE leave-one-match-out and 2 Metrica zero-shot; clean/degraded are averaged within each Metrica match first. MacroF1 is the tie-breaker; count error does not replace the main metric. All 9 fold/scenario rows per configuration are in `results/wave3_development.json`.

| # | Family | Decoder | Mean spellF1 | Mean macroF1 | Max abs count error |
|---:|---|---|---:|---:|---:|
| 1 | hybrid_tree25 | centered2 | 0.615394 | 0.873323 | 0.4168 |
| 2 | hybrid | centered2 | 0.613449 | 0.873529 | 0.3973 |
| 3 | hybrid_tree25 | team05_dead2 | 0.612492 | 0.872935 | 0.3875 |
| 4 | hybrid | team05_dead2 | 0.610887 | 0.872795 | 0.3836 |
| 5 | hybrid_tree25 | cost05 | 0.610821 | 0.872949 | 0.3855 |
| 6 | setseq | centered2 | 0.610136 | 0.870886 | 0.3855 |
| 7 | hybrid | cost05 | 0.609823 | 0.872685 | 0.3796 |
| 8 | setseq | team05_dead2 | 0.608521 | 0.870031 | 0.3699 |
| 9 | setseq | cost05 | 0.607472 | 0.870203 | 0.3620 |
| 10 | hybrid | cost2 | 0.606614 | 0.871993 | 0.4677 |
| 11 | hybrid_tree50 | team05_dead2 | 0.605708 | 0.869219 | 0.3777 |
| 12 | hybrid | team2_dead05 | 0.605204 | 0.871776 | 0.4638 |
| 13 | hybrid_tree50 | cost05 | 0.604500 | 0.868877 | 0.3659 |
| 14 | hybrid_tree25 | cost2 | 0.603684 | 0.872127 | 0.4736 |
| 15 | hybrid_tree25 | team2_dead05 | 0.602476 | 0.871954 | 0.4716 |
| 16 | hybrid_tree50 | centered2 | 0.601856 | 0.868013 | 0.4677 |
| 17 | setseq | cost2 | 0.600078 | 0.869230 | 0.4579 |
| 18 | setseq | team2_dead05 | 0.598662 | 0.869317 | 0.4481 |
| 19 | hybrid_tree50 | cost2 | 0.593213 | 0.866667 | 0.4912 |
| 20 | hybrid_tree50 | team2_dead05 | 0.590878 | 0.866401 | 0.4736 |
| 21 | hybrid | cost6 | 0.568256 | 0.868403 | 0.5205 |
| 22 | setseq | cost6 | 0.555624 | 0.866024 | 0.5264 |
| 23 | hybrid_tree25 | cost6 | 0.549742 | 0.866851 | 0.5499 |
| 24 | hybrid_visibility | centered2 | 0.534255 | 0.813619 | 0.3679 |
| 25 | hybrid_visibility | cost2 | 0.529367 | 0.812371 | 0.3992 |
| 26 | hybrid_tree50 | cost6 | 0.524832 | 0.858872 | 0.5969 |
| 27 | hybrid_visibility | team2_dead05 | 0.509922 | 0.810043 | 0.3131 |
| 28 | hybrid_tree75 | centered2 | 0.480214 | 0.737072 | 0.4501 |
| 29 | hybrid_tree75 | cost2 | 0.473812 | 0.735027 | 0.4618 |
| 30 | hybrid_visibility | cost6 | 0.469769 | 0.808332 | 0.5616 |
| 31 | hybrid_tree75 | team2_dead05 | 0.454333 | 0.731328 | 0.3562 |
| 32 | hybrid_visibility | team05_dead2 | 0.430402 | 0.807795 | 0.7050 |
| 33 | hybrid_visibility | cost05 | 0.422888 | 0.806285 | 0.7391 |
| 34 | hybrid_tree75 | cost6 | 0.413357 | 0.737900 | 0.6204 |
| 35 | feature_gru | cost05 | 0.398470 | 0.804888 | 0.5362 |
| 36 | feature_gru | team05_dead2 | 0.397944 | 0.804727 | 0.5421 |
| 37 | feature_gru | centered2 | 0.395107 | 0.806296 | 0.5695 |
| 38 | feature_gru | team2_dead05 | 0.392972 | 0.805419 | 0.6125 |
| 39 | feature_gru | cost2 | 0.391968 | 0.805451 | 0.6184 |
| 40 | hybrid_tree75 | team05_dead2 | 0.380696 | 0.731778 | 0.5652 |
| 41 | hybrid_tree75 | cost05 | 0.371290 | 0.727037 | 0.5963 |
| 42 | feature_gru | cost6 | 0.371048 | 0.804392 | 0.6556 |
| 43 | all_features | cost6 | 0.358719 | 0.769908 | 0.5540 |
| 44 | all_features | cost2 | 0.341599 | 0.751008 | 0.5458 |
| 45 | control_symmetric | cost2 | 0.338520 | 0.663530 | 0.4286 |
| 46 | all_features | centered2 | 0.334167 | 0.746242 | 0.7932 |
| 47 | control_symmetric | centered2 | 0.334117 | 0.653292 | 0.4881 |
| 48 | control | cost2 | 0.327532 | 0.644256 | 0.3699 |
| 49 | control | centered2 | 0.323729 | 0.638883 | 0.6475 |
| 50 | all_features | team2_dead05 | 0.311170 | 0.745424 | 1.2136 |
| 51 | anchor | cost6 | 0.306517 | 0.625730 | 0.5300 |
| 52 | control_symmetric | cost6 | 0.306346 | 0.676986 | 0.6771 |
| 53 | control | cost6 | 0.299590 | 0.660926 | 0.6673 |
| 54 | control_symmetric | team2_dead05 | 0.277299 | 0.630987 | 0.5932 |
| 55 | control | team2_dead05 | 0.276458 | 0.610189 | 0.7112 |
| 56 | anchor | centered2 | 0.273167 | 0.614331 | 0.4317 |
| 57 | anchor | cost2 | 0.273167 | 0.614331 | 0.4317 |
| 58 | anchor | team2_dead05 | 0.242419 | 0.609601 | 0.5528 |
| 59 | control_symmetric | team05_dead2 | 0.217938 | 0.643836 | 1.5864 |
| 60 | all_features | team05_dead2 | 0.208926 | 0.727328 | 2.5729 |
| 61 | control | team05_dead2 | 0.206338 | 0.616641 | 2.0407 |
| 62 | all_features | cost05 | 0.196705 | 0.722837 | 3.1797 |
| 63 | control_symmetric | cost05 | 0.190868 | 0.617068 | 1.8508 |
| 64 | control | cost05 | 0.189084 | 0.594735 | 2.2237 |
| 65 | anchor | team05_dead2 | 0.148035 | 0.594204 | 2.1429 |
| 66 | anchor | cost05 | 0.139193 | 0.590849 | 2.2267 |

## Reproduction and checks

- 34 prepared packs: 21 IDSSE (7 matches × 3 rounds), 12 Metrica (3 matches × 4 conditions), 1 target. Features, blocks, y/y25 alignment and unknown-team masks pass.
- 14 raw IDSSE metadata/tracking files, 2,622,686,298 bytes in total, verified against the size and LFS SHA-256/Git object inventory of PySport. Raw data stays local.
- The target team mapping is not guessed from the 24/22 slot order. At least 20 votes and 99% purity per sampled present slot; 43 present known slots and 3 empty sampled slots. 37,651 finite team distances match exactly; 37 unavailable distances have the same availability mask and are not imputed as valid observations. The audit covers only sampled graded frames at 5 fps; it does not claim that every 25-fps team identity was verified.
- 6 regression tests pass: exhaustive optimum for six decoders; reset/zero-probability/empty blocks; visibility fusion; permutation invariance and mirror involution; control feature preservation and per-half velocity; missing team-distance equivalence.
- The three tree families pass synthetic fit/predict, constant H/A/D, output normalisation, label-free test, input immutability and boundary reset.
- The three neural families pass synthetic train/predict, an independent deterministic refit and a checkpoint round trip. cuDNN GRU on Windows produced deterministic values but crashed at process teardown (−1073740791). CPU and native CUDA finished cleanly; the campaign chose single-thread CPU without suppressing errors or reducing models/epochs.
- 36 development model fits completed with exit code 0. Four target CSVs pass the official validator. The candidate was chosen without target labels.
- The selected candidate was retrained independently on all 5 development IDSSE matches and the Metrica anchor with the same seed. All 199,319 target rows and the CSV SHA-256 are identical.
- The selection hash is identical between the holdout and public reports. Code/inputs/checkpoints were verified before both phases.

Candidate CSV: `out/candidates/selected.csv`. SHA-256 `337734b6bf9b70bda6c9a16b549e39e3c90bdbd53fc6073c140ab07cd06b6096`.

Aggregate evidence safe to share: `results/wave3_{development,holdout,public,execution,verification}.json`. Checkpoints, raw data, probabilities and per-frame CSVs are ignored by Git.

## Follow-up experiments still worth doing (not run)

1. Reproduce Jeremy's original checkpoint/recipe on a stable PyTorch backend before claiming that this combination beats Jeremy. The local adapter is smaller and not a substitute for his original result.
2. A pre-registered seed ensemble on the same data to measure the stability of the 0.002 mixture gain and the centered decoder, rather than picking a seed from the public window.
3. Add holdout datasets whose possession/dead-ball definitions are close to the challenge. Off-pitch cues and count rates differ by domain; cross-provider validation is needed before an aggressive count-aware decoder.
4. The raw target bundle with metadata, so the input pipeline can be regenerated without another member's cache. A verified nearest-distance mapping does not replace raw-metadata provenance.

All 66 configurations in [PLAN.md](PLAN.md) were run. The list above is a research decision after the results, not unfinished parts of the campaign. There was no further training/tuning based on the public diagnostic.

Audit of Jeremy's source: [JEREMY_REVIEW.md](JEREMY_REVIEW.md). Reproduction commands: [README.md](README.md).
