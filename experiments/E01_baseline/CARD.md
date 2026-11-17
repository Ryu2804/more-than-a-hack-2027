# E01: Official baseline (nearest player + majority vote)

| | |
|---|---|
| **Folder** | [`experiments/E01_baseline/`](./) |
| **Builds on** | E00 |
| **Status** | done |
| **Verdict** | reference floor |

## Question
How does the official baseline behave, and how do spellF1 and macroF1 disagree when only the smoothing window changes?

## Hypothesis
Wider smoothing reduces flicker and raises spellF1, at some cost to per-frame accuracy.

## Design
- **Method:** nearest tracked player to the ball (metres) gives a team code per frame; majority vote over ±w frames, w ∈ {0, 25, 50, 125}.
- **Single factor:** window w. Everything else held constant.
- **Evaluation:** public 5-minute GT as a pipeline check only. Full-match numbers quoted from `EVALUATION.md`.

## Metric and decision rule
No selection. The organisers' published floor is smooth50.

## Result
Public window (official scorer, 17 GT possessions):

| Window | spellF1 | macroF1 | Predicted possessions |
|---|---:|---:|---:|
| raw | 0.037838 | 0.597227 | 131 |
| ±25 | 0.286792 | 0.619991 | 36 |
| ±50 | 0.325581 | 0.620733 | 26 |
| ±125 | 0.424242 | 0.581185 | 16 |

Full match, smooth50: spellF1 0.2433 (published by the organisers).

## Verdict
Kept as the reference floor. macroF1 moves ±0.04 while spellF1 moves about five times as much, and the two metrics pick different windows. The public window favours ±125, so choosing w from it would be wrong.

## Threats to validity
One factor, one match; the public ranking of windows does not transfer to the full match.

## Evidence
[`E01_baseline.ipynb`](E01_baseline.ipynb), [`evaluation/results/baseline_*.csv`](../../evaluation/results/).
