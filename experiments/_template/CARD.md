# Exx: <name>

| | |
|---|---|
| **Folder** | `experiments/...` |
| **Builds on** | Exx (control comes from there) |
| **Status** | planned / running / done |
| **Verdict** | promoted / not promoted / rejected / benchmark only |

## Question
One sentence. What decision does this experiment inform?

## Hypothesis
Written before fitting. What should change, and in which direction?

## Design
- **Unit of replication:** (match, not frame). Count of independent units.
- **Conditions:** control vs. treatment(s); what is held constant.
- **Blocking / nuisance:** provider, corruption round, half, seed; how each is handled.
- **Split:** which matches train, select, hold out. Leakage guards.
- **Size:** configurations, seeds, fits, evaluations.

## Metric and decision rule
Primary metric, aggregation order, and the promotion rule (minimum gain, win count, interval),
fixed before results are seen.

## Result
Numbers from the results file only, with the comparison against the control and its interval.

## Verdict
Apply the rule as written. Do not move the threshold after seeing the result.

## Threats to validity
Exposure of holdout data, proxy labels, multiple comparisons, pseudoreplication, compute mismatch.

## Evidence
Links to PLAN.md, RESULTS.md, results JSON, hashes.
