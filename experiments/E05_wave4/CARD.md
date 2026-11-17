# E05: Wave4, PP-TCN across three providers

| | |
|---|---|
| **Folder** | [`experiments/E05_wave4/`](./) |
| **Builds on** | E04; adapts Ryu branch `5eaf3e5` |
| **Status** | done |
| **Verdict** | selected `pp_gru25/no_shift`; control for E06 |

## Question
Do Ryu's per-player PP-TCN and cross-provider training (adding SkillCorner) improve external generalisation?

## Hypothesis
More providers and a permutation-invariant per-player branch matter more than the choice of sequence model.

## Design
- **Data added:** SkillCorner open data (MIT), phases-of-play proxy labels.
- **Training:** 19 matches (5 IDSSE, 14 SkillCorner), six grouped folds; all corruption rounds of a match leave together; normalisation and rate statistics from the training fold only.
- **Development:** 23 scenarios from 21 matches; Metrica 1/2 zero-shot.
- **Size:** 64 configurations, 1,472 development evaluations, 63 seed fits.
- **Architecture:** residual TCN, 6 blocks, kernel 5, 64 channels, dilations (1, 2, 4, 8, 16, 1), GroupNorm, two heads (dead, team); player-set MLP with attention + max pooling.

## Metric and decision rule
**Equal-provider mean spellF1**: average conditions within each match, then matches within each provider, then the three providers equally. Ties macroF1, family, decoder. Freeze before holdout and public.

## Result
Selected `pp_gru25/no_shift`: 75% new PP-TCN + 25% frozen wave3 GRU; rate-calibrated decoder without dead-prior shift. Development equal-provider spellF1 0.656490.

Post-freeze public diagnostics: selected 0.772414 (12 possessions vs 17), pure new PP-TCN 0.772414, previous wave3 0.575610, shipped Ryu checkpoint 0.760000.

## Verdict
Frozen; becomes the control `old_gru25/wave4` in E06.

## Threats to validity
Shipped Ryu weights had seen all IDSSE/SkillCorner matches and are not valid held-out predictions; they were not used for selection. SkillCorner labels are phase proxies.

## Evidence
[`wave4/PLAN.md`](PLAN.md), [`wave4/RESULTS.md`](RESULTS.md), [`wave4/reference/SOURCE.md`](reference/SOURCE.md), [`results/wave4_*.json`](results/).
