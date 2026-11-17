# E02: Wave1, rule and learned families on Metrica

| | |
|---|---|
| **Folder** | [`experiments/E02_wave1/`](./) |
| **Builds on** | E01 |
| **Status** | done |
| **Verdict** | selected `m01_hysteresis/r15`; superseded by E03 |

## Question
Which of eight method families generalises best on external matches with proxy labels?

## Hypothesis
Temporal consistency (hysteresis, confirmation) beats per-frame nearest-player labelling under occlusion.

## Design
- **Data:** Metrica sample Games 1/2 (event-derived proxy labels), each in original and synthetically occluded condition.
- **Split:** two leave-one-game-out folds. Game 3 held out and opened only after the freeze.
- **Size:** 8 families, 24 configurations, 96 external evaluations.
- **Unit:** game. Only two development games, so the search is adaptive.

## Metric and decision rule
Rank by mean spellF1 over the four fold × condition cells; tie-break macroF1, then method, then config. Highest rank is frozen.

## Result
Selected `m01_hysteresis/r15` (radius 1.5 m, confirm 0.4 s, gap→dead 8 s).

| Evaluation | spellF1 |
|---|---:|
| Mean Games 1/2, clean + occluded | 0.460517 |
| Game 3 original | 0.474935 |
| Game 3 added occlusion | 0.374534 |

Post-freeze public diagnostic: 0.442105.

## Verdict
Frozen as wave1 candidate; superseded by E03.

## Threats to validity
Two development games; one holdout game; proxy labels differ from the challenge definition. Occlusion clearly lowers spellF1.

## Evidence
[`results/wave1_metrics.json`](results/wave1_metrics.json), [`E03_wave2/RESULTS.md`](../E03_wave2/RESULTS.md) (wave1/2 history). Candidate CSV SHA-256 `6a022089…`.
