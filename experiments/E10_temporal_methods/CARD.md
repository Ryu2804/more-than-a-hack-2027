# E10: Duration-aware and consensus decoders

| | |
|---|---|
| **Folder** | [`experiments/E10_temporal_methods/`](./) |
| **Builds on** | E09 (control short75 + SetSeq25 / dead45, 0.615458) |
| **Status** | done |
| **Verdict** | rejected (primary −0.009620) |

## Question
Can explicit duration modelling or boundary consensus fix the control's possession under-count (3,061 GT vs 2,208 predicted spells) without new training?

## Hypothesis
Primary: an HSMM with explicit H/A/D durations (strength 0.5, switch bias 0) beats the dead45 decoder.

## Design
- **No neural fits.** Same cached OOF probabilities, fixed 75/25 fusion.
- **Methods:** HSMM durations, consensus change-point (reduced switch penalty where models agree, plus contact), ordered boundary DP.
- **Size:** 19 configurations (1 control + 6 per method) × 7 matches = 133 evaluations. Duration priors learned from each fold's 6 training matches.

## Metric and decision rule
Gain ≥ 0.005 vs 0.615458, wins ≥ 5/7, paired-bootstrap 95% lower bound > 0. If the primary fails, keep the control; do not substitute the best raw.

## Result
Primary HSMM: 0.605838, Δ −0.009620, CI [−0.019067, −0.002950], wins 1/7.
Best exploratory `consensus/alpha0.75_contact0.5`: 0.616970, Δ +0.001512, CI [−0.000155, 0.003510], 5/7.

## Verdict
**Rejected.** Control retained. Consensus is a small, unconfirmed improvement.

## Threats to validity
Seven exposed matches; intervals not corrected for multiple comparisons.

## Evidence
[`temporal_methods/PLAN.md`](PLAN.md), [`temporal_methods/RESULTS.md`](RESULTS.md), [`results/temporal_methods_*.json`](results/).
