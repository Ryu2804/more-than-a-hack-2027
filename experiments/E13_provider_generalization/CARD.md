# E13: Provider generalisation (source-only transfer)

| | |
|---|---|
| **Folder** | [`experiments/E13_provider_generalization/`](./) |
| **Builds on** | E12 (event sampler and short loss as base recipe) |
| **Status** | done |
| **Verdict** | rejected (primary −0.010486) |

## Question
Do regularisers improve transfer to a provider never seen in training, as a check on overfitting risk?

## Hypothesis
Primary: modality dropout (drop pooled-player channels or the whole PP branch) reduces provider-specific reliance and improves transfer.

## Design
- **Two transfer directions:** train 7 IDSSE → test 20 SkillCorner clean; train 20 SkillCorner → test 7 IDSSE.
- **Five recipes**, fresh PP-TCN (135,219 parameters), 3 seeds × 500 updates: 30 seed fits. Normalisation, event indices and rate calibration from the source provider only.
- **Leakage guard:** label-free target inference; 300 artifacts sealed before any target metric was computed. No competition labels accessed.

## Metric and decision rule
Gain ≥ 0.01 on the equal mean of the two target providers, neither provider below −0.02, and provider-stratified paired-bootstrap 95% lower bound > 0 (seed 20261120).

## Result
Control 0.540864. Primary `modality_dropout` 0.530378: Δ −0.010486, CI [−0.016208, −0.004824] (IDSSE target −0.021938, SkillCorner target +0.000967). RDrop 0.540882 (practically equal to control).

## Verdict
**Rejected.** Control retained. No regulariser improved cross-provider transfer.

## Threats to validity
All 27 external matches previously exposed: no fresh blind cohort, so this cannot certify freedom from overfitting. Numbers are not comparable with the IDSSE-ensemble numbers of E08–E12 (different protocol).

## Evidence
[`provider_generalization/PLAN.md`](PLAN.md), [`provider_generalization/RESULTS.md`](RESULTS.md), [`EXPOSURE.md`](EXPOSURE.md), [`results/provider_generalization_*.json`](results/).
