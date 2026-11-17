# E08: Shared-split benchmark of model families (IDSSE)

| | |
|---|---|
| **Folder** | [`experiments/E08_fair_idsse/`](./) |
| **Builds on** | E06 (decoder dead45 fixed) |
| **Status** | done |
| **Verdict** | benchmark only; no promotion |

## Question
Under identical data, labels, split and decoder, how do the team's model families compare?

## Hypothesis
The short-loss PP-TCN family leads; the old HGB family trails neural models even when retrained fairly.

## Design
- **Split:** 7 IDSSE matches, leave-one-match-out (each fold trains on the other 6).
- **Arms:** short-loss PP-TCN (Farras-style), SetSeq (Jeremy-style adapter), PP-TCN (Ryu-style), HGB symmetric (Farras wave2-style).
- **Budget fixed in advance:** seeds 0/1/2, 500 updates/seed, batch 16, window 384 for every neural arm; 63 fresh neural seed fits, 7 HGB fits. Decoder wave5 `dead45`, not tuned per model or fold.
- **Unit:** match; paired bootstrap over matches (5,000 draws, seed 20261114).

## Metric and decision rule
Mean official spellF1 over 7 held-out matches; macroF1 secondary. No candidate selection or promotion from this table.

## Result
| Family | spellF1 | macroF1 | Count error |
|---|---:|---:|---:|
| Short-loss PP-TCN | 0.595755 | 0.902935 | −23.98% |
| SetSeq | 0.578602 | 0.896519 | −32.51% |
| PP-TCN | 0.571090 | 0.900344 | −33.82% |
| HGB symmetric | 0.315719 | 0.771451 | +1.13% |

Short-loss vs PP-TCN: +0.024666, CI [0.018060, 0.032109], 7/7. Short-loss vs SetSeq: +0.017153, CI [−0.003553, 0.038950], 5/7 (not separated).

## Verdict
Benchmark result only. Short-loss PP-TCN leads on the mean but is not clearly separated from SetSeq. Becomes the control for E09.

## Threats to validity
Adapted families, not each branch's exact final pipeline. Reduced budget (500 vs 1,500 updates). Seven matches only.

## Evidence
[`fair_idsse/PLAN.md`](PLAN.md), [`fair_idsse/RESULTS.md`](RESULTS.md), [`results/fair_idsse_*.json`](results/).
