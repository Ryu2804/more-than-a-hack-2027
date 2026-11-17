# E03: Wave2, gradient boosting + Viterbi

| | |
|---|---|
| **Folder** | [`experiments/E03_wave2/`](./) |
| **Builds on** | E02 (frozen wave1 candidate as reference) |
| **Status** | done |
| **Verdict** | selected `w08_symmetric_boost/cost6`; superseded by E04 |

## Question
Do learned tree models with a Viterbi switch cost beat the wave1 rule?

## Hypothesis
A classifier on control features (team distances, margin, visibility, gap age/remaining) plus a constant-cost Viterbi produces fewer, longer and more correct possessions than hysteresis.

## Design
- **Same protocol as E02:** Metrica Games 1/2, LOGO folds, original + occluded; Game 3 holdout after freeze.
- **Size:** 16 families, 48 configurations, 192 external evaluations.
- **Treatment of note:** H/A symmetric augmentation (swap features, margin and labels).

## Metric and decision rule
Same ranking as E02. `run.py all` refuses to overwrite a frozen selection.

## Result
Selected `w08_symmetric_boost/cost6`: HistGradientBoosting with H/A swap augmentation, Viterbi switch cost 6.

| Evaluation | Wave1 | Wave2 |
|---|---:|---:|
| Mean Games 1/2, clean + occluded | 0.460517 | 0.517438 |
| Game 3 original | 0.474935 | 0.599664 |
| Game 3 added occlusion | 0.374534 | 0.433982 |

Post-freeze public diagnostic 0.460606. The highest public diagnostic (0.560000, config `w14_feature_ablation/control`, dev 0.500969) was **not** adopted because it was not the external choice.

Negative results: shape-only features, GaussianNB, confirmation gate, learned transition logistic and dead-ball bridging did not beat the selected config.

## Verdict
Frozen; superseded by E04.

## Threats to validity
72 configurations across E02–E03 reuse the same two development games. Game 3 is one match with proxy labels.

## Evidence
[`results/wave2_metrics.json`](results/wave2_metrics.json), [`RESULTS.md`](RESULTS.md), [`results/relocation_verification.json`](results/relocation_verification.json). Candidate CSV SHA-256 `45e03662…`.
