# Final method: wave5 `short_gru25/dead45`

The submitted method is wave5 `short_gru25/dead45` from [E06](../experiments/E06_wave5/). The submitted file is `carried_by_admin.csv` (SHA-256 `850e6d25…`): this method retrained from scratch. This document is the reference description of the method; [final_method.json](final_method.json) holds the same facts as data, and the technical report reads them from there. Primary/control decisions inside other experiments apply only to those campaigns.

## Identity

- Method ID: `farras_wave5_short_gru25_dead45`.
- Implementation release: commit [`7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60), branch `farras`.
- Campaign selection SHA-256: `6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`.

| CSV | SHA-256 | Public spellF1 / macroF1 | Role |
|---|---|---:|---|
| `carried_by_admin.csv` (copy: `submission/artifacts/e06_retrained.csv`) | `850e6d25d96ea051d0f746548ab59de9b4c04fdededca53fff16fbe9c2053b5c` | 0.772414 / 0.901002 | **Submitted.** Retrained from scratch by [`E06_wave5.ipynb`](../experiments/E06_wave5/E06_wave5.ipynb) ([receipt](../experiments/E06_wave5/results/wave5_retrain_verification.json)) |
| `submission/artifacts/farras_wave5.csv` | `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8` | 0.772414 / 0.902059 | Frozen campaign candidate; the selection evidence describes it. Reference only |

Both have 199,319 rows (`frame,label`, BAR-ATH tactical timeline at 25 fps) and agree on 99.52% of graded frames. Neither is ground truth. Neither is wave6 or the wave2 gradient-boosting model of the old consolidation table. Code on other branches belongs to its own experiments ([branch map](BRANCHES.md)).

## Pipeline

1. **Observations.** Automatic tracking of the target is turned into team positions/distances and observation features. The model works offline at 5 fps; predictions are expanded to the 25-fps timeline. Video, human event labels of the target, Smart Tagging and homography are not inputs. For the submitted file, the E06 notebook rebuilds the target cache and features from the raw match bundle in `data/`.
2. **Short-loss PP-TCN (75%).** A three-seed PP-TCN ensemble. The state loss is doubled for complete live spells shorter than 4 s. Durations are measured on the full record within each half, so crop fragments and unknown runs are not up-weighted.
3. **Set-sequence BiGRU (25%).** The wave3 hybrid model: a player-set encoder and a bidirectional GRU. The joint H/A/D probabilities of the two components are mixed 75/25 and renormalised. The frozen candidate reused the wave3 GRU checkpoint; the submitted file retrains it from scratch with the same recipe. The mixture contains no Metrica tree anchor and no auxiliary boundary head.
4. **Decoder `dead45`.** Dead threshold 0.45, minimum dead duration 1 s, distance gate 0.35 m, kappa 0.05, tau 3, merge 1 s, rate scale 1. The H/A dynamic programme is reset in every live interval; the rate calibration counts possessions after merging. Rate statistics come from the training matches, not from the public GT.
5. **No target-specific rules.** No dead-prior shift, forced dead balls or kickoff correction. Half boundaries come from the competition metadata. Centred features and the bidirectional GRU make the method offline, not real-time.

The full decoder configuration, including inactive parameters, is in the JSON. `fixed_lambda=2.0` is inactive: the mode used is `lambda_mode="rate"`.

## Training and selection

The PP-TCN is trained on 19 matches: 5 IDSSE and 14 SkillCorner. Development used six grouped folds; all corruption rounds of a match leave the training set together. Normalisation and decoder statistics come from the training fold only. Recipe: seeds 0/1/2, 1,500 optimiser updates per seed, batch 16, crop of 384 samples, AdamW learning rate 0.002, weight decay 0.01, OneCycle, dropout 0.15; 64 channels, residual temporal convolutions with kernel 5 and dilations 1/2/4/8/16/1.

The BiGRU is trained on the five IDSSE development matches only, not on all 19: seed 2027, 8 epochs, window 300, batch 16, hidden/width 32, two bidirectional GRU layers. The two components therefore differ in training set and number of seeds.

Metrica Games 1/2 (clean and degraded) are used for development and selection, Game 3 for evaluation; none is training data. The development objective is the equal mean over three providers, after averaging conditions within each match. Out of 96 configurations, a replacement for wave4 had to gain at least 0.01, win at least two of three providers, and lose no more than 0.03 on any provider. Wave5 passed that rule before the public diagnostic was looked at. The 63 seed fits of the campaign cover three model families and all folds; inference uses three.

External data and label sources:

- IDSSE: Bassek et al., CC BY 4.0, [paper](https://doi.org/10.1038/s41597-025-04505-y), [data](https://doi.org/10.6084/m9.figshare.28196177). Possession labels from the dataset provider.
- SkillCorner [open data](https://github.com/SkillCorner/opendata), MIT. Provider phases of play, used as proxy labels.
- Metrica [sample-data](https://github.com/metrica-sports/sample-data): event-derived proxy labels, evaluation only. No explicit licence was found. Individual annotators are not documented.

## Recorded numbers

| Evaluation | Metric | Value | Meaning |
|---|---|---:|---|
| Wave5 external development (frozen candidate) | Equal-provider spellF1 | 0.670099305497763 | Selection objective, not a competition score |
| Development vs wave4 control | Delta spellF1 | +0.013608981205591952 | Same-protocol comparison |
| Competition public GT, submitted file | SpellF1 | 0.7724137931034483 | 5-minute diagnostic, not leaderboard/private |
| Competition public GT, submitted file | MacroF1 | 0.9010016369961317 | Secondary metric |
| Competition public GT, frozen candidate | SpellF1 / MacroF1 | 0.7724137931034483 / 0.9020586271571484 | Reference only |
| Full match (private) | SpellF1 and macroF1 | Unknown | No private score exists |

The official spellF1 is the mean one-to-one possession F1 at IoU 0.3/0.4/0.5/0.6/0.7, including all short H/A spells; macroF1 is the mean frame F1 over H/A/D. IDSSE benchmark numbers come from a different protocol and are not comparable with the public score.

Baseline for comparison: the official smooth50 scores public spellF1 0.325581 and published full-match spellF1 0.2433 (different cohorts). Wave6 scores 0.786207 on the public window but was not adopted: its development gain missed the promotion threshold.

## Limitations

- No path from competition GT into training, inference or selection was found on the audited code. This is not a certificate of being leakage-free.
- All external holdout matches and the public window had been seen by earlier campaigns. The holdout and stress results are adaptive estimates, not a fresh blind test.
- Wave5 is slightly worse than wave4 on the paired ball-drop stress test; precision and long-spell recall remain open problems.
- The submitted CSV is produced end-to-end (public training data → training → raw match bundle → CSV) by the E06 notebook. Retraining on other hardware or library versions can change the weights slightly, so a rerun is expected to agree closely with the submitted file but not byte for byte. External development and holdout scores were measured on the frozen candidate only.

From any checkout, `python -B submission/export.py` validates and copies the submitted CSV without data, caches, weights or GT.

## Evidence

On `main`:

- Recipe and pre-registered selection rule: [E06 PLAN.md](../experiments/E06_wave5/PLAN.md); results including negative ones: [E06 RESULTS.md](../experiments/E06_wave5/RESULTS.md).
- Selection receipt: [wave5_verification.json](../experiments/E06_wave5/results/wave5_verification.json); public audit: [wave5_public.json](../experiments/E06_wave5/results/wave5_public.json); retrain receipt: [wave5_retrain_verification.json](../experiments/E06_wave5/results/wave5_retrain_verification.json).
- Code: PP-TCN [models.py](../experiments/E05_wave4/models.py), short-loss [training_models.py](../experiments/E06_wave5/training_models.py), GRU [models.py](../experiments/E04_wave3/models.py), [fusion](../experiments/E06_wave5/adaptive_fusion.py), [decoder](../experiments/E06_wave5/segment_decoder.py).
- [Data/code attribution](../experiments/E05_wave4/reference/SOURCE.md), [exposure audit](../experiments/E13_provider_generalization/EXPOSURE.md).

The same files at the implementation release, where the campaigns still lived under `experiments/exp_farras/`: [runtime manifest](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/method.json), [wave5 folder](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5) ([relocation note](../experiments/REGISTRY.md#relocation-note)).

## Attribution

The PP-TCN and its features come from Ryu's research; the player-set/sequence model from Jeremy's; the short-possession loss, the mixture and the selected decoder from Farras's campaigns. Older write-ups on the `ryu` and `rasya` branches describe earlier variants, not this method.
