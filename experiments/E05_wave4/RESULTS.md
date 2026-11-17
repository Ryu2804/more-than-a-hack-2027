# Wave4: cross-provider PP-TCN with Ryu's reference

Candidate frozen through external validation: **`pp_gru25/no_shift`**. The public score improves over wave3, but the pure PP-TCN matches its public spellF1. The strongest evidence of improvement comes from cross-provider data and the per-player representation, not from proof that the GRU mixture is required. The private/full-match score is unknown.

## Public: four fixed candidates after the freeze

Official scorer, `bootstrap_n=0`, public GT window of 7,500 frames with 17 GT possessions. No 64-configuration search on the public GT.

| Candidate | SpellF1 | MacroF1 | Predicted possessions |
|---|---:|---:|---:|
| selected | 0.772414 | 0.895984 | 12 |
| new_pp_tcn | 0.772414 | 0.899359 | 12 |
| previous_wave3 | 0.575610 | 0.712844 | 24 |
| shipped_ryu | 0.760000 | 0.919932 | 13 |

`selected` means 75% probabilities from the new PP-TCN + 25% wave3 hybrid GRU, not 75% weights from Ryu's checkpoint that had already seen the holdout. `new_pp_tcn` is the pure new model with the `ryu_rate` decoder; `previous_wave3` is the frozen wave3 candidate, not a retuned wave3. The two new candidates have the same public spellF1 even though their macroF1 and per-frame labels differ. The choice was not changed after seeing the public macroF1.

The selected candidate predicts 12 versus 17 GT possessions: a 29.41% under-count. Its macroF1 is below Ryu's shipped adapter. A spellF1 difference on one small window is not evidence of statistical or private superiority.

## Audit of Ryu's branch

Source pinned at [`5eaf3e595e91fa0502aac48037d0cae8c356ec6f`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/5eaf3e595e91fa0502aac48037d0cae8c356ec6f). `017_multi_match/NOTES.md` reports about 0.747 public for the full-mix player-set TCN; the old technical document's 0.422 is not the latest result. Notes on a grouped 6-fold, 30-match evaluation report about 0.661 without the player set and 0.679 with it. Filtered numbers that drop short possessions are not used as the main metric of this campaign.

Architecture adopted: pooled player features and two team views, a residual TCN with 6 blocks, kernel 5, 64 channels, dilations `(1,2,4,8,16,1)`, GroupNorm, and two heads (dead/team). The player set adds a per-player MLP, attention pooling and max pooling. The decoder scores H/A switches with a ball-distance gate, rate calibration from training, merging of live slivers, and a separate dead overlay.

The shipped `weights_nometrica` checkpoint turned out to be trained on **all 27 matches: 7 IDSSE + 20 SkillCorner**. It is therefore forbidden as a held-out prediction for IDSSE/SkillCorner and was not used to select among the 64 configurations. The shipped reference is evaluated only on Metrica 3 and the target. The checkpoint's report metadata gives a provider-balanced rate of 3.5828524 possessions/minute and a dead share of 0.4574683; these are not training statistics of the new model, nor an assumption that 0.747 holds exactly for this checkpoint.

The shipped score in the table is an **adapter**, not a byte-exact reproduction of Ryu's original CSV. The differences: target inputs come from the local cache, the unknown team slots 21/22/43 are masked, prediction/decoding use the 5-fps grid and labels are repeated to 25 fps, the rate picks the nearest lambda on the grid instead of log interpolation, and there is no target-specific forced dead interval at kickoff. The 25-fps feature code and Ryu's three state dicts are original. The five source snapshots are unedited; provenance in [reference/SOURCE.md](reference/SOURCE.md).

## Data, split and protocol

- 20 SkillCorner matches downloaded from the public upstream with size and LFS SHA-256 validation. Raw tracking 1,823,448,331 bytes; 25-fps timeline 3,112,002 frames, 591,139 samples at 5 fps inside the half intervals.
- New training uses only 19 matches: 5 IDSSE development and 14 SkillCorner development. IDSSE has three corruption rounds; provider, match, round and then crop are sampled in balance during training.
- 6 grouped folds exclude every round of the evaluated match. Training matches per fold: 15/15/16/16/16/17. Metrica 1/2 clean/degraded is zero-shot stress for the TCN and GRU, not a training source for either model.
- Objective: mean spellF1 per match, clean/degraded averaged within a match, then the mean per provider with equal weight for IDSSE/SkillCorner/Metrica. MacroF1 tie-breaker. Count error is recorded, not used as a hidden objective.
- 23 development scenarios come from 21 matches: 5 IDSSE, 14 SkillCorner, 2 Metrica with two conditions each. 64 configurations give 1,472 evaluations, not 1,472 independent models.
- Campaign holdout: 2 IDSSE, 6 SkillCorner and Metrica 3 clean/degraded, i.e. 9 unique matches in 10 scenarios. IDSSE/Metrica 3 were exposed in earlier campaigns; the 6 SkillCorner matches are new for Farras but were used in Ryu's research. Not called globally blind.
- All new training and inference is local. Target GT is not used for training or selection. The registry locks source, inputs, prior checkpoints, split, recipe and device before fitting; the selection is hashed before holdout and public.

Rules, match names, decoders and recipe before results: [PLAN.md](PLAN.md). The earlier wave3 development mean is not compared directly with the wave4 mean because the dataset and weighting changed.

## Development results and ablations

Three new architectures: pooled TCN, player-set TCN, and player set + 7 control channels. Ensemble of 3 seeds, 1,500 optimiser steps/seed, batch 16, crop 384, AdamW learning rate 0.002 / weight decay 0.01, OneCycle, dropout 0.15. Each fold/full model is trained separately. Ball masking also masks the ball-related PP channels and all 7 control channels so the model has no leak path.

Eight probability families are combined with eight decoders. The table shows the best configuration of each family and one matched `ryu_rate` control for the selected family. All 64 configurations and per-match metrics are in [wave4_development.json](results/wave4_development.json).

| Family | Decoder | Mean spellF1 | Mean macroF1 | IDSSE spellF1 | SkillCorner spellF1 | Metrica spellF1 |
|---|---|---:|---:|---:|---:|---:|
| plain_tcn | no_shift | 0.609125 | 0.875965 | 0.516401 | 0.683777 | 0.627198 |
| pp_tcn | no_shift | 0.649605 | 0.881666 | 0.570895 | 0.707571 | 0.670350 |
| control_tcn | no_shift | 0.650108 | 0.881234 | 0.570460 | 0.707524 | 0.672340 |
| pp_gru25 | no_shift | 0.656490 | 0.884522 | 0.583755 | 0.709044 | 0.676673 |
| pp_gru50 | no_shift | 0.650717 | 0.883483 | 0.586995 | 0.676695 | 0.688462 |
| pp_gru75 | no_shift | 0.633341 | 0.874851 | 0.577774 | 0.626968 | 0.695281 |
| pp_tree25 | no_shift | 0.646579 | 0.881659 | 0.567203 | 0.705694 | 0.666839 |
| prior_combined | no_shift | 0.614330 | 0.865600 | 0.565785 | 0.592999 | 0.684204 |
| pp_gru25 | ryu_rate | 0.653885 | 0.883764 | 0.585623 | 0.704171 | 0.671859 |

The player set beats the pooled TCN on 21/21 development matches with the matched `no_shift` decoder. Adding the 7 control channels to the TCN gives only a small gain; there is no basis for calling it a strong improvement. The 25% GRU mixture gives a small development gain over the PP-TCN, below the 0.01 adoption margin mentioned in Ryu's research. This campaign uses the fixed ranking rule in the PLAN; it does not claim that the mixture passed Ryu's adoption rule or a significance test.

`no_shift` means the dead prior is not forced towards 0.407; it does not remove the dead head or dead overlay. Other parameters: gate 0.35 m, kappa 0.05, merge 1 s, dead threshold 0.5, minimum dead run 2 s, tau 3 s, rate scale 1. Final training-only provider-balanced rate 3.6428176 possessions/minute, dead share 0.4608204. Every best family chose `no_shift`, but the full target GT is not available to prove that this prior decision generalises.

Feature corrections: speeds are computed at the real 5 fps; 2-s rolling and 1-s median reset per half; each player's opponent distance uses that player's opposing team, not the view's opposing-team shortcut that could include the player itself. Missing observations are not turned into valid distance-gate evidence. Calibration evaluates the full lambda grid, without assuming the possession count is monotonic after decoding.

## Holdout after selection

Mean spellF1 per provider. The Metrica column is two conditions of the same match, not two independent matches.

| Candidate | IDSSE (2) | SkillCorner (6) | Metrica 3 clean/degraded |
|---|---:|---:|---:|
| selected | 0.644282 | 0.676204 | 0.657178 |
| new_pp_tcn | 0.632643 | 0.672262 | 0.639989 |
| previous_wave3 | 0.597654 | 0.585117 | 0.701784 |
| shipped_ryu | not valid as holdout | not valid as holdout | 0.635642 |

On the 8 IDSSE/SkillCorner holdout matches, the selected candidate wins 7/8 against wave3 and 7/8 against the default PP-TCN. The mean per-match gain over wave3 is 0.079972, but this comparison changes the training data together with the architecture/decoder; it is not an isolated effect of the GRU mixture. The worst absolute SkillCorner count error drops from 48.93% (wave3) to 6.33%; IDSSE is still 29.73%. The selected candidate's worst development count error is still 44.62%, so it does not meet the 30% count-error bound in every case.

Metrica 3 **regresses compared with wave3**, in both the clean and the degraded condition. The new provider average does not remove that domain failure. Details of the 10 scenarios:

| Match/condition | Selected | PP-TCN default | Wave3 frozen |
|---|---:|---:|---:|
| J03WQQ/r0 | 0.654913 | 0.636805 | 0.657420 |
| J03WR9/r0 | 0.633651 | 0.628481 | 0.537888 |
| metrica3/clean | 0.671637 | 0.653722 | 0.740230 |
| metrica3/r0 | 0.642718 | 0.626256 | 0.663339 |
| skillcorner_2010085/clean | 0.667532 | 0.669623 | 0.568276 |
| skillcorner_2011166/clean | 0.700952 | 0.698252 | 0.562940 |
| skillcorner_2013725/clean | 0.683980 | 0.676819 | 0.595556 |
| skillcorner_2015213/clean | 0.685333 | 0.675433 | 0.594564 |
| skillcorner_2016236/clean | 0.660929 | 0.658699 | 0.595930 |
| skillcorner_2017461/clean | 0.658494 | 0.654745 | 0.593438 |

Shipped Ryu on Metrica 3: clean 0.651803, degraded 0.619481. Only those two rows are valid as an external reference, because every IDSSE/SkillCorner match was a training source of that checkpoint. All 32 rows and the adapter audit: [wave4_holdout.json](results/wave4_holdout.json).

## Runtime and reproduction actually run

- 11 regression tests pass: exhaustive DP, minimum dead run/half reset, live sliver, non-monotonic lambda, known/visible ball gate, partition validation, conditional fusion, PP opponent/permutation, unit/half reset, masked control/PP, and rejection of the shipped reference on in-sample providers.
- A smoke test of actual fitting/prediction/checkpoint/refit passed for all three architectures. CUDA Conv1d exit 0; wave3's local CUDA GRU shutdown problem does not affect TCN training. The GRU prior is prepared on CPU in a separate process.
- Development/full training campaign: 21 ensemble fits, **63 seed fits** with the full recipe, 94,500 optimiser steps; CUDA device, exit 0. Recorded time 4,082.453 s.
- After the freeze, the selected PP-TCN was retrained from seed on the same 19 matches (221.735 s). The frozen GRU component was not retrained in wave4; its checkpoint/inputs are hashed and wave3's refit evidence still holds. The mixture and decoder produce **199,319 target labels identical byte for byte** with the original candidate.
- The wave3 CSV was also reproduced with the same historical hash before being used as a comparison. The selection hash stays the same after holdout/public.
- Ryu's original pinned feature code emits `RuntimeWarning: Mean of empty slice` and `Degrees of freedom <= 0 for slice.` on missing observations; the warnings are not hidden. The adapter still produces finite probabilities and the four CSVs pass the scorer. No warning-free runtime is claimed.

Local selected-candidate CSV: `E05_wave4/out/candidates/selected.csv`.

CSV SHA-256:

`ddd75416e7929183954d27191b11ddab1ac049fc0e703f802db2209884635e9a`

Frozen selection SHA-256:

`b5269672e86cce891c7ce28054ce17a6acfa4c4979e11236757212d908acdd05`

Aggregates safe to share: [execution](results/wave4_execution.json), [verification](results/wave4_verification.json), [public](results/wave4_public.json), plus development and holdout above. Raw data, per-frame probabilities/labels, model weights, caches and submission CSVs stay in ignored folders.

## License and limits of the conclusions

[SkillCorner open data](https://github.com/SkillCorner/opendata) is MIT licensed. IDSSE: [Bassek et al., Scientific Data 12, 195 (2025)](https://doi.org/10.1038/s41597-025-04505-y), [data CC BY 4.0](https://doi.org/10.6084/m9.figshare.28196177). Ryu's provenance and source credit are kept.

The new TCN and the hybrid GRU in the selected candidate are not trained on Metrica. The `pp_tree25` and `prior_combined` families use the old anchor tree trained on Metrica; the Metrica 1/2 evaluation of the anchor excludes the evaluated game, but the unclear license of the anchor's training data remains. Metrica is used for zero-shot evaluation and is not automatically considered explicitly licensed.

External data uses event/phase proxies, not identical to the competition's corrected-GT definition. The 64-configuration search uses the same development data, augmentation does not guarantee noise equivalent to the target, and several holdout matches had been exposed in other campaigns or research. The model uses future temporal context and runs offline. The public and SkillCorner-holdout improvements are real on this evaluation; full/private superiority is not verified. The next priorities are fresh external validation and short-possession/occlusion robustness without dropping short GT from the metric.
