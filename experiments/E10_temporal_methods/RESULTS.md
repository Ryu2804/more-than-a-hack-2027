# Temporal methods results

## Decision

The control stays `short75% + SetSeq25% / dead45`, mean official IDSSE spellF1 **0.6154575861548992**. The primary fixed before evaluation, `duration/strength0.5_bias0`, gives 0.6058379034594431. Difference −0.009619682695456211, paired-bootstrap 95% CI [−0.019066773114166515, −0.002949900067270631], wins 1/7. It fails the promotion rule; the control is not replaced.

The best raw exploratory configuration is `consensus/alpha0.75_contact0.5`: spellF1 **0.6169697856726231**, macroF1 0.909851793, up 0.0015121995177236971, wins 5/7. The interval of the difference [−0.00015497175080957811, 0.0035096818464089275] still crosses 0 and the gain is below 0.005. This is a small improvement that is not yet robust, not evidence that the new model is better. The alpha0.5/contact0.5 variant wins 6/7 with a nominally positive interval, but its gain of 0.0014639193262721006 is also below 0.005 and it was picked from many exploratory variants.

No new public/private score or target submission. This is adaptive research on seven matches that have been seen repeatedly, not a fresh blind estimate. The intervals are not corrected for multiple comparisons or the selection history.

## New methods and all results

All use the same cached OOF neural predictions and the fixed 75/25 probability fusion. Only the duration priors learn, from the six training matches of each fold. Consensus lowers the switch penalty at directional changes on which both models agree, plus observed contact. Boundary relocation optimises the boundary locations jointly while keeping the ordered state runs and their number. The HSMM models H/A/D durations with an unbounded geometric tail, without a hard dead gate or a target spell count. The formulas are frozen in [PLAN.md](PLAN.md).

| Configuration | spellF1 | macroF1 | Difference | Wins/7 |
|---|---:|---:|---:|---:|
| `consensus/alpha0.75_contact0.5` | 0.616970 | 0.909852 | +0.001512 | 5 |
| `consensus/alpha0.5_contact0.5` | 0.616922 | 0.909863 | +0.001464 | 6 |
| `consensus/alpha0.75_contact0` | 0.616689 | 0.909815 | +0.001231 | 4 |
| `control/dead45` | 0.615458 | 0.909587 | +0.000000 | 0 |
| `consensus/alpha0.25_contact0.5` | 0.615177 | 0.909588 | -0.000281 | 2 |
| `consensus/alpha0.5_contact0` | 0.614820 | 0.909644 | -0.000637 | 2 |
| `consensus/alpha0.25_contact0` | 0.614806 | 0.909572 | -0.000652 | 0 |
| `boundaries/radius2_penalty0` | 0.614800 | 0.909956 | -0.000657 | 2 |
| `boundaries/radius2_penalty0.05` | 0.614339 | 0.909967 | -0.001118 | 2 |
| `boundaries/radius5_penalty0` | 0.613149 | 0.909675 | -0.002309 | 1 |
| `duration/strength0.25_bias0` | 0.613077 | 0.909092 | -0.002381 | 1 |
| `boundaries/radius5_penalty0.05` | 0.613017 | 0.909686 | -0.002441 | 1 |
| `boundaries/radius10_penalty0` | 0.612572 | 0.909494 | -0.002886 | 1 |
| `boundaries/radius10_penalty0.05` | 0.612572 | 0.909490 | -0.002886 | 1 |
| `duration/strength0.25_bias-0.5` | 0.612076 | 0.909371 | -0.003381 | 2 |
| `duration/strength0.5_bias-0.5` | 0.609884 | 0.908987 | -0.005574 | 2 |
| `duration/strength0.5_bias0` | 0.605838 | 0.908576 | -0.009620 | 1 |
| `duration/strength1_bias-0.5` | 0.590225 | 0.907336 | -0.025233 | 0 |
| `duration/strength1_bias0` | 0.583344 | 0.906548 | -0.032113 | 0 |

The primary metric is mean official spellF1 across IoU 0.3/0.4/0.5/0.6/0.7, over all H/A spells including short ones. MacroF1 is secondary. Wins are counted per unique match against the control; ties are not wins.

## Per match

| Match | Control | Primary HSMM | Best consensus |
|---|---:|---:|---:|
| J03WMX | 0.546667 | 0.543396 | 0.553431 |
| J03WN1 | 0.566628 | 0.563088 | 0.568009 |
| J03WOH | 0.651744 | 0.645655 | 0.654440 |
| J03WOY | 0.641802 | 0.604941 | 0.643000 |
| J03WPY | 0.618994 | 0.619799 | 0.617827 |
| J03WQQ | 0.654496 | 0.642735 | 0.655435 |
| J03WR9 | 0.627871 | 0.621250 | 0.626646 |

## Diagnosis after the run

The recall below uses IoU 0.75 as a diagnostic, not the primary metric. The ground truth is the same for every method. Predictions are pooled H/A spells; count error is the mean relative error per match.

| Method | Predicted/GT spells | Mean count error | Matched < 4 s / 1,621 | Matched 4 to < 15 s / 1,058 | Matched ≥ 15 s / 382 |
|---|---:|---:|---:|---:|---:|
| Control | 2208/3061 | −0.273891 | 230 | 539 | 301 |
| Primary HSMM | 2050/3061 | −0.324976 | 203 | 513 | 303 |
| Best consensus | 2218/3061 | −0.270864 | 233 | 546 | 299 |
| Boundary radius2/penalty0 | 2208/3061 | −0.273891 | 222 | 542 | 301 |

The primary HSMM actually produces fewer spells and lower short recall; long durations improve slightly. [INFERENCE] At this weight the duration prior adds smoothing that does not fit the under-segmentation problem. Every HSMM strength tested loses to the control, so the strength is not raised or adopted on the basis of the hypothesis alone.

Boundary relocation keeps the number of spells by design, but every radius/penalty tested lowers spellF1. [INFERENCE] Raising the frame log-probability with a fixed label order is not enough to improve an IoU objective, especially when short spells are already lost. Consensus adds 10 pooled spells and slightly more short/medium recall; a measurable but small change.

## Verification and provenance

47 semantic tests pass, including exhaustive tiny-sequence optimality for the HSMM and the boundary DP, censored training intervals, an unforced 257-frame duration tail, half resets, contact agreement, immutable inputs and frozen-decoder parity at constant weights. The smoke test of the 19 settings on 96 real observations from two halves passes without metric evaluation.

The full evaluation of 19 settings × 7 folds = 133 metric rows took 29.39 s, excluding semantic tests/smoke, pre-registration and the independent rebuild. Zero new neural fits; seven supervised duration-prior fits. This is decoder research on cached predictions, not a comparison of training cost or a new architecture.

An independent rebuild in another process finished with 133 identical decoded arrays and seven identical refitted duration priors. Every checks/evaluate/rebuild process exited 0.

Source/PLAN/input manifest SHA-256: `38829cd0f0f48a1ffffa2d4dd0cca57c14dcc7f684a95b58eed0e697b0076b75`. The predecessor's decoded control must be byte-identical per match, not just score the same. Full results and per-match bootstrap inputs: `results/temporal_methods_results.json`; protocol/checks/rebuild proof are shared with the same prefix. Raw training-duration lists stay in the ignored `out/priors/` and are not published.
