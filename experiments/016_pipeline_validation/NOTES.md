# 016 pipeline validation on other matches (the first real measurement)

**Problem addressed:** until now nothing could be measured: the public window has 17 possessions and the full ground truth is hidden. The Metrica sample games come with event data, so the WHOLE pipeline (ball evidence + TCN + Viterbi + dead ball + lambda calibration) can be run on a held-out game and scored with the challenge's own metric code (`evaluation/possession_metrics.py`) against that game's event-derived labels. Predictions for each game come from a model trained only on the other two games (leave-one-game-out), lambda is calibrated to the other games' possession rate, and `harness.py` / `decoder.py` are exactly the code the submission uses. Labels are our derivation from events, not the challenge ground truth. Training runs on the marimo sandbox (`remote.py`); local code only decodes and scores.

**Headline (3 held-out matches, ~320 possessions each):**

| mean over games | baseline_smooth50 | pipeline as first shipped | with adopted choices |
|---|---|---|---|
| spellF1 | 0.429 | 0.561 | 0.598-0.604 |
| macroF1 | 0.767 | 0.796 | 0.80 |
| team accuracy | 0.879 | 0.922 | 0.93 |
| transition F1 (+/-1 s) | 0.222 | 0.481 | 0.51-0.52 |

**Adoption rule fixed before looking (one-at-a-time sweeps):** adopt a change only if mean spellF1 over the 3 games improves by >= 0.01 and it wins on >= 2 of 3 games.

**Adopted:** control-gate radius 0.35 m (from 1.0 m; +0.014 with the team head on, 3/3; monotone down to ~0.25-0.5 m, matching the literature's 0.5-1 m range); team head fed the real ball channel instead of hidden (+0.013, 3/3); merging H/A runs shorter than 0.5 s after calibration (+0.017, 3/3). **Not adopted (below the rule):** kappa 0.01 vs 0.02 (+0.006), soft-control evidence (+0.007), dead-ball threshold/shift (+0.006 to +0.010, 1-2/3 games), tau, minimum stoppage length, a possession-rate scale of 0.9 (+0.006).

**Findings worth keeping:** shape evidence is essential (kappa 0: -0.069); lambda calibration is robust (+/-15% on the rate target moves spellF1 by <0.01); underestimating the possession count costs more than overestimating it (0.7x: -0.028, 1.3x: -0.016); calibrating lambda to the rate needs a grid reaching below 1 (it saturated at 2 for small gate radii and under-counted possessions). **Timing:** on held-out games the signed offset of possession changes is ~0 (median 0.00 s, optimum at zero shift; 58-61% of true changes matched within 0.5 s, 68-77% within 1 s), so there is no systematic early/late bias to correct; the early flips seen on the public window therefore come from the challenge's own definition (controlled possession), not a bug.

**Training-variant batch (3 Metrica games, same rule, `remote.py`):** hiding the ball in only 30% of training crops (+0.018, 3/3) and input-noise augmentation 0.5 sd (+0.018, 3/3) pass; more seeds (+0.005), longer receptive field (+0.002), 4000 steps (0.000), width 128 (+0.003), mask 0.7 (+0.004) do not.
