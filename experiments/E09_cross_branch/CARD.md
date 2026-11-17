# E09: Cross-branch cue combination

| | |
|---|---|
| **Folder** | [`experiments/E09_cross_branch/`](./) |
| **Builds on** | E08 (control short-loss PP-TCN/dead45, 0.595755) |
| **Status** | done |
| **Verdict** | research primary passed; best raw kept as exploratory |

## Question
Do cues from every branch (Jeremy contact features, Rasya touch-memory/dead cues, Ryu PP encoder, Farras short loss) improve on the short-loss PP-TCN when combined?

## Hypothesis
Primary, declared before fitting: `triple_combined/hysteresis` beats `short/dead45`.

## Design
- **Split and budget as E08:** 7 IDSSE LOSO folds, 3 seeds × 500 updates; 63 fresh seed fits, 28 probability/decoder settings, 196 per-match evaluations.
- **Primary mixture:** 50% short-loss PP-TCN, 25% SetSeq, 25% contextual TCN (contact + touch-memory/dead-cue features); hysteresis low 0.35 / high 0.50.

## Metric and decision rule
Promote (for research) if gain ≥ 0.005, wins ≥ 5/7 matches and paired-bootstrap 95% lower bound > 0 (5,000 draws, seed 20261115).

## Result
| Pipeline | spellF1 | macroF1 |
|---|---:|---:|
| Control short-loss PP-TCN/dead45 | 0.595755 | 0.902935 |
| Primary `triple_combined/hysteresis` | 0.609030 | 0.909290 |
| Best raw: short75 + SetSeq25 / dead45 | 0.615458 | 0.909587 |

Primary gain +0.013274, CI [0.005245, 0.021057], 6/7. Best raw gain +0.019702, CI [0.015308, 0.024763], 7/7.

## Verdict
Primary passes the research rule. The simpler two-model mixture scores higher but was not the primary, so it is reported as exploratory and becomes the control for E10. Combining every feature is not automatically best.

## Threats to validity
Seven previously exposed matches; best-of-28 is not an unbiased estimate. Research promotion only, no submission change.

## Evidence
[`cross_branch/PLAN.md`](PLAN.md), [`cross_branch/RESULTS.md`](RESULTS.md), [`cross_branch/SOURCES.md`](SOURCES.md), [`results/cross_branch_*.json`](results/).
