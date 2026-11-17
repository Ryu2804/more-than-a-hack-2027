# E07: Wave6, fusion and decoder alternatives

| | |
|---|---|
| **Folder** | [`experiments/E07_wave6/`](./) |
| **Builds on** | E06 (control wave5, dev 0.670099) |
| **Status** | done |
| **Verdict** | not promoted (gain 0.004707 < 0.005) |

## Question
Can the stress weaknesses of wave5 (extra false possessions, lost long spells) be fixed by fusion or decoding alone, without new training?

## Hypothesis
Conservative mixtures, separated dead/team heads, and dead-ball hysteresis reduce false possessions under ball loss.

## Design
- **No new training.** Frozen wave5 probabilities and inputs; audit of six remote branches informs the alternatives.
- **Size:** six method paths (blends, visibility weighting, heads, occlusion cost, dead hysteresis, short guard), 36 alternatives + 1 control, 851 development evaluations on 23 scenarios / 21 matches.
- **Aggregation:** as E05/E06.

## Metric and decision rule
Eligible if gain ≥ 0.005 over wave5, wins ≥ 2/3 providers, no provider drops > 0.01, and the paired match-bootstrap 95% lower bound > 0 (5,000 draws, seed 20261114, matches resampled within provider).

## Result
Best: `m05_dead_hysteresis/low35_high50` (accept a p_dead > 0.35 run only if it contains a sample > 0.50). Development 0.674807 vs 0.670099: gain 0.004707, 95% CI [0.003412, 0.006096].

| Provider (dev) | Wave5 | Best alt. | Δ |
|---|---:|---:|---:|
| IDSSE | 0.612109 | 0.609661 | −0.002448 |
| SkillCorner | 0.709337 | 0.718859 | +0.009522 |
| Metrica | 0.688852 | 0.695901 | +0.007049 |

Holdout: wins 7/10 ordinary scenarios and 6/6 stress. Post-freeze public diagnostic 0.786207 (macroF1 0.925369).

## Verdict
**Not promoted.** The gain misses the pre-set 0.005 minimum by 0.000293, and the threshold was not lowered after seeing the result. Wave5 stays selected. The higher public score is not a selection signal.

## Threats to validity
36-way search not corrected for multiplicity; all data previously exposed.

## Evidence
[`wave6/PLAN.md`](PLAN.md), [`wave6/RESULTS.md`](RESULTS.md), [`wave6/BRANCH_REVIEW.md`](BRANCH_REVIEW.md), [`results/wave6_*.json`](results/).
