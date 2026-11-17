# E00: Exploratory data analysis

| | |
|---|---|
| **Folder** | [`experiments/E00_eda/`](./) |
| **Builds on** | none |
| **Status** | done |
| **Verdict** | descriptive; shaped all later designs |

## Question
What do the target tracking, Smart Tagging and homography look like, and which properties should constrain the model design?

## Hypothesis
None (descriptive). The public GT is used only to check frame alignment, never to set thresholds.

## Design
- **Unit:** one match (the target), 199,319 frames, 158,857 graded. Frame-level statistics are descriptive, not inferential.
- **Inputs:** tactical ATD (46 track slots), Smart Tagging (638 intervals), broadcast homography. Tactical homography was not yet downloaded when the notebook ran.

## Metric and decision rule
No decision rule. Outputs are summary statistics and figures.

## Result
- Ball visible on 59.3% of graded frames. 150 gaps: median 10.7 s, P90 33.8 s, max 221 s; 85.2% of missing frames lie in gaps > 10 s.
- P(ball lost within 1 s | visible) = 3.6%: loss is a persistent state, not per-frame noise.
- Tracked players per graded frame: mean 20.5, max 25. 2,218 tracklets, median 13.6 s, P90 201 s.
- Nearest player to the ball: median 1.9 m. Home/away distance margin < 2 m on 32.5% of visible-ball frames.
- Smart Tagging `Team` on "defending" codes names the defending side. Naive agreement with a nearest-player baseline 53.7%, 63.4% after flipping.
- Broadcast homography valid on 65.8% of frames; broadcast not synchronised to tactical.
- Public GT: 7,500 frames, 17 possessions, mean 14.2 s.

## Verdict
Design implications adopted downstream: long temporal context; team-level or set features rather than per-slot features; proximity evidence only when the ball is visible and close; Smart Tagging and broadcast not used as inputs.

## Threats to validity
Single match; statistics describe this recording only. 17 public possessions are far too few for model selection.

## Evidence
Notebook [`E00_eda.ipynb`](E00_eda.ipynb) (English; outputs cleared for a rerun; the original Indonesian run with outputs is at commit `efdb82e`, `experiments/00_eda.ipynb`). Summary table: Table I of [`docs/report/report.typ`](../../docs/report/report.typ).
