# E12: Event-learning extensions

| | |
|---|---|
| **Folder** | [`experiments/E12_event_learning/`](./) |
| **Builds on** | E11 (strong control `event_control/consensus`, 0.619563) |
| **Status** | done |
| **Verdict** | rejected (primary −0.000371) |

## Question
Starting from the useful event sampler, do six further training extensions beat the strong control?

## Hypothesis
Primary: balanced event replacement (`balanced_replace/consensus`) beats the strong control.

## Design
- **Six recipes**, all with 50% event-centred crops, 3 seeds × 500 updates, 7 LOSO folds: 126 fresh seed fits, 40 settings / 280 evaluations, 42 saved-model replays.

## Metric and decision rule
Gain ≥ 0.005, wins ≥ 5/7, paired-bootstrap 95% lower bound > 0 (5,000 draws, seed 20261118). No best-raw fallback.

## Result
Primary `balanced_replace/consensus`: 0.619192, Δ −0.000371, CI [−0.003314, 0.002354], 3/7 + 1 tie.
Best exploratory `contact_replace/consensus`: 0.624577, Δ +0.005014, CI [−0.000212, 0.011266], 5/7.
`ranking_replace/dead45`: 0.621512, 7/7 vs strong control but Δ +0.001949.

## Verdict
**Rejected.** Strong research control retained. The best exploratory recipe passes the gain and win thresholds but its interval crosses 0, and it was not the primary.

## Threats to validity
Seven exposed matches; nominal intervals; comparisons with different decoders confound model and decoder effects.

## Evidence
[`event_learning/PLAN.md`](PLAN.md), [`event_learning/RESULTS.md`](RESULTS.md), [`results/event_learning_*.json`](results/).
