# Team final method: Farras wave5

The team chose **Farras wave5 `short_gru25/dead45`** as the submission candidate and as the method described in the technical report. The choice of method is final; the submission package is not yet complete. This document and [final_method.json](final_method.json) are the reference for the report on every branch. Primary/control decisions inside older experiment reports apply only to those campaigns.

## Locked identity

- Method ID: `farras_wave5_short_gru25_dead45`.
- Release source: commit [`7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60), originating branch `farras`.
- [Algorithmic prediction CSV](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/artifacts/farras_wave5.csv): `submission/artifacts/farras_wave5.csv`, 199,319 rows, `frame,label`, BAR-ATH tactical timeline at 25 fps. The same file is published at the repository root as `submission.csv`.
- CSV SHA-256: `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.
- Campaign selection SHA-256: `6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`.

This CSV is not ground truth, not wave6, and not the wave2 HGB that once appeared in the consolidation table. Research code on other branches still belongs to its original experiment; aligning the documentation does not turn those algorithms into wave5. [Branch and archive map](BRANCHES.md).

## Pipeline to describe

1. Observations come from automatic tracking of the target, prepared into a cache of team positions/distances and observation features. The model works offline at 5 fps and its predictions are expanded to the 25-fps timeline. Video, human target event labels, Smart Tagging and homography are not inference inputs on this candidate path. The public package does not yet regenerate the target cache end-to-end from the raw bundle.
2. The main component is a three-seed PP-TCN ensemble with the wave5 short-possession loss. The state loss is doubled for complete live spells shorter than 4 seconds. Durations are measured on the full record within each half; crop fragments and unknown runs do not receive a spurious short weight.
3. The joint H/A/D probabilities are mixed **75% short-TCN and 25% frozen wave3 hybrid SetSequence/GRU**, then renormalised. The GRU is Farras's own model, not Ryu's checkpoint and not a mixture with the Metrica tree anchor. It uses a player-set encoder and a bidirectional GRU. Do not call this mixture a boundary-head ensemble: the short-TCN candidate does not train a new auxiliary boundary head.
4. The `dead45` decoder uses dead threshold 0.45, minimum dead duration 1 second, distance gate 0.35 m, kappa 0.05, tau 3, merge 1 second and rate scale 1. The H/A dynamic programme is reset in every live interval; the rate calibration counts possessions after merging. Rate statistics come from training, not from the number of public GT possessions.
5. The dead-prior shift is off. The selected candidate has no target-specific forced-dead or kickoff correction. Official half boundaries come from the competition metadata, not from hand-made possession labels. Centred features and the bidirectional GRU make the method offline, not real-time.

The full decoder configuration, including inactive parameters, is in the JSON. `fixed_lambda=2.0` is not an effective fixed lambda: the mode used is `lambda_mode="rate"`.

## Training and selection

The short-TCN component was trained on 19 matches: 5 IDSSE and 14 SkillCorner. Development used six grouped folds; all corruption rounds of a match leave the training set together. Normalisation and decoder statistics come from the training fold only. Short-TCN recipe: seeds 0/1/2, 1,500 optimiser updates per seed, batch 16, crop of 384 samples, AdamW learning rate 0.002, weight decay 0.01, OneCycle, dropout 0.15. The architecture uses 64 channels and residual temporal convolutions with kernel 5 and dilations 1/2/4/8/16/1.

The frozen GRU comes from wave3 and was trained on the five IDSSE development matches, not on all 19 PP-TCN matches. Its configuration: seed 2027, 8 epochs, window 300, batch 16, hidden/width 32 and two bidirectional GRU layers. Do not state that every component shares the same training set or number of seeds.

Metrica Games 1/2 clean/degraded are used for external development and selection, and Game 3 for evaluation; they are not training data for the neural components of the selected method. The development objective is the equal mean over three providers after aggregating conditions within each match. Out of 96 configurations, a replacement for wave4 had to gain at least 0.01, win at least two of three providers, and lose no more than 0.03 on any provider. Wave5 passed that rule before the public diagnostic. The figure of 63 seed fits is the campaign total over three model families and folds/final, not the number of models used at inference by the final candidate.

External data and label sources:

- IDSSE: dataset by Bassek et al., CC BY 4.0, [paper](https://doi.org/10.1038/s41597-025-04505-y), [data](https://doi.org/10.6084/m9.figshare.28196177). Possession labels from the dataset provider.
- SkillCorner [open data](https://github.com/SkillCorner/opendata), MIT; provider phases of play used as external proxy labels.
- Metrica [sample-data](https://github.com/metrica-sports/sample-data): event-derived proxy labels for validation. The source audit found no explicit license; do not claim that all data use is license-clean. Individual annotator identities are not recorded in the repository evidence and must not be invented.

## Numbers allowed in the report

| Evaluation | Metric | Value | Meaning |
|---|---|---:|---|
| Wave5 external development | Equal-provider spellF1 | 0.670099305497763 | Selection objective, not a competition score |
| Development vs wave4 control | Delta spellF1 | +0.013608981205591952 | Same-protocol comparison |
| Competition public GT | SpellF1 | 0.7724137931034483 | 5-minute diagnostic, not leaderboard/private |
| Competition public GT | MacroF1 | 0.9020586271571484 | Secondary metric |
| Full-match/private competition | SpellF1 and macroF1 | Unknown | No private model score exists |

The official spellF1 is the mean one-to-one temporal possession F1 at IoU 0.3/0.4/0.5/0.6/0.7, including all short H/A spells. MacroF1 is the mean frame F1 over H/A/D. Do not call accuracy the main metric, and do not rank IDSSE benchmark numbers together with the public score.

If the report needs a baseline: the official smooth50 has public spellF1 0.325581 and published full-match spellF1 0.2433. They are different cohorts. The wave6 public result of 0.786207 is a research alternative that was not adopted; its development gain missed the promotion threshold. That number is not a score of this final method.

## Limits on claims and readiness

- No flow of competition GT into training, inference or selection was found on the audited paths. That is not a certificate of being leakage-free, nor a guarantee that the whole experiment history is free of public-informed adaptation.
- Public diagnostics of earlier campaigns were already known. All external holdout matches have been exposed historically; the paired stress test is not new matches. Do not call it a fresh blind test, an unbiased estimate or proof of no overfitting.
- Wave5 is slightly worse on the paired ball-drop stress test; precision and long-spell recall remain problems. There is no evidence of superiority across all domains or of an exact final ranking of all branches.
- Refitting the new component on the prepared inputs once produced an identical target CSV. That is not a raw-bundle reproduction: frozen GRU priors and the target cache are still reused. The hashes lock the identity of the cache; they do not prove its raw origin.
- The public package can export the frozen CSV without data/cache/weights/GT; the official raw bundle → identical CSV path is not complete. **Do not write that the package meets every reproduction requirement of RULES.md section 3 or is ready for official submission.** Method selection and package compliance are different statuses.

From any checkout, `python -B submission/export.py` validates and copies the published CSV. No training or scoring was run to align this documentation.

## Evidence sources for the Typst author

Read [final_method.json](final_method.json) as data rather than reconstructing numbers from chat or taking the maximum over all reports. A `.typ` file in `docs/` can load it with `#let method = json("final_method.json")`. Private values that are `null` must be written as unknown, not converted to zero.

Code and report references at the same release:

- [Runtime manifest](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/method.json).
- [Wave5 recipe and selection rule](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/PLAN.md).
- [Full wave5 results, including negative results](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/RESULTS.md).
- [Selection/refit evidence](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/results/wave5_verification.json), [public audit](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/results/wave5_public.json).
- [PP-TCN configuration](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/models.py), [short-loss training](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/training_models.py), [GRU](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave3/models.py).
- [Separating the neural prior from the anchor](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/prepare_prior.py), [fusion](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/adaptive_fusion.py), [decoder](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/segment_decoder.py).
- [Data/code attribution](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/reference/SOURCE.md), [exposure audit](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/provider_generalization/EXPOSURE.md).

These links point at the release commit, where the campaigns still lived under `experiments/exp_farras/`. On `main` the same files are in `experiments/E05_wave4/`, `experiments/E06_wave5/` and their `results/` folders ([relocation note](../experiments/REGISTRY.md#relocation-note)).

Ryu's work provided the PP-TCN/feature foundation; Jeremy's work provided the player-set/sequence model reference; the integration of the short loss, the mixture and the selected decoder happened in Farras's campaigns. Do not copy old technical write-ups by Ryu/Rasya as the description of the final implementation. This documentation preserves each member's source and results; it does not claim the whole method was built by one person.
