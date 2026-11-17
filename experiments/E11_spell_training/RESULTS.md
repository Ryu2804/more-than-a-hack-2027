# Spell training results

## Decision

The primary `event_replace25/dead45` reaches mean official IDSSE spellF1 **0.6189411362728311**, versus control 0.6154575861548992. Gain 0.0034835501179317934, paired match-bootstrap 95% CI [0.0006699738275977982, 0.006503930284240906], wins 5/7. The difference signal is positive, but the gain does not reach the 0.005 requirement frozen before the run. **The primary is not promoted; the control is kept.** The rule was not changed after seeing the score.

Best raw exploratory: `event_replace25/consensus`, spellF1 **0.619562844991005**, macroF1 **0.9099829750309686**. Gain 0.004105258836105574 over the dead45 control, CI [0.0010056306129022005, 0.007612324713658802], wins 5/7. Its gain is also below 0.005, and this configuration is not the primary.

The previous best, 0.6169697856726231, is another exploratory consensus variant; the new best raw is 0.0025930593183819406 higher. To isolate the model effect from the decoder, the new best is compared with a control using the same consensus decoder (alpha 0.5 / contact 0.5): gain 0.0026413395098334735, CI [−0.0005110195422673357, 0.00601190972269252], wins 4/7. This interval still crosses 0. Do not attribute the whole gain over dead45 to the new training alone.

All intervals use 5,000 paired bootstrap draws, seed 20261117, unit = unique match. The seven matches and the hypotheses had been exposed; the nominal intervals do not correct for the adaptive history or multiple comparisons. This is IDSSE research, not a public/private competition score. No target evaluation or new submission.

## What changed

- Spell-balanced TCN: the team BCE is weighted by the inverse square root of the duration of complete H/A spells, normalised by the weighted live mass; ordinary dead BCE. Runs touching a half endpoint or an unknown label are not considered complete.
- Event TCN: the original short loss stays, but with 50% probability the crop within an already-chosen match is replaced by a crop around a complete H/A spell < 4 seconds. Jitter/clamp keeps the crop within one half and the event inside the crop. Match selection stays balanced; held-out labels are not used.
- Dual-scale TCN: combines the original-dilation Conv5 and a dilation-1 convolution with the same kernel/bias before the GELU. No extra parameters; more convolution compute. Loss and sampling stay the original short recipe.

All three are trained fresh on the other six matches in each LOSO fold, 3 seeds / 500 updates / batch 16 / crop 384. The same 29 global channels, 48 × 10 PP features, width 64 and **135,219 parameters per network**. Total 63 seed fits / 31,500 updates. No predecessor weights are loaded. The SetSeq and short-loss control probabilities are only reused from the audited held-out cache, not retrained.

## All configurations

All numbers are per-match means over the seven IDSSE folds. `replace25` means short 50% + SetSeq 25% + new model 25%; `base75` means 75% of the control probability mixture plus 25% new model. `pure` means the new model alone. Consensus uses a fixed alpha 0.5 / contact 0.5 rather than searching for the best decoder again. Deltas are relative to `control/dead45`.

| Configuration | spellF1 | macroF1 | Delta | Wins/7 |
|---|---:|---:|---:|---:|
| `event_replace25/consensus` | 0.619563 | 0.909983 | +0.004105 | 5 |
| `event_replace25/dead45` | 0.618941 | 0.909976 | +0.003484 | 5 |
| `control/consensus` | 0.616922 | 0.909863 | +0.001464 | 6 |
| `control/dead45` | 0.615458 | 0.909587 | +0.000000 | 0 |
| `spell_replace25/dead45` | 0.615418 | 0.909749 | -0.000040 | 4 |
| `spell_replace25/consensus` | 0.615025 | 0.909745 | -0.000433 | 3 |
| `event_base75/consensus` | 0.615014 | 0.908300 | -0.000443 | 5 |
| `dual_replace25/consensus` | 0.613749 | 0.909313 | -0.001709 | 3 |
| `event_base75/dead45` | 0.613662 | 0.908173 | -0.001795 | 3 |
| `spell_base75/consensus` | 0.613384 | 0.908575 | -0.002074 | 2 |
| `dual_replace25/dead45` | 0.612194 | 0.909044 | -0.003263 | 1 |
| `spell_base75/dead45` | 0.611257 | 0.908367 | -0.004200 | 0 |
| `dual_base75/consensus` | 0.608949 | 0.907944 | -0.006509 | 0 |
| `event_pure/consensus` | 0.608846 | 0.904182 | -0.006611 | 2 |
| `event_pure/dead45` | 0.608023 | 0.904088 | -0.007435 | 2 |
| `dual_base75/dead45` | 0.607290 | 0.907672 | -0.008168 | 0 |
| `spell_pure/dead45` | 0.604291 | 0.902495 | -0.011167 | 0 |
| `spell_pure/consensus` | 0.604094 | 0.902565 | -0.011363 | 1 |
| `dual_pure/dead45` | 0.587528 | 0.900496 | -0.027930 | 0 |
| `dual_pure/consensus` | 0.587513 | 0.900674 | -0.027944 | 0 |

## Model ablation without the ensemble

The secondary comparison below was computed after the run and does not change the primary decision. The reference is the original short-loss PP-TCN/dead45 from fair_idsse, **not** the 0.615458 ensemble control. Folds, inputs, decoder, seeds and update budget are comparable. It shows training effects that could be hidden if every pure model were only compared with the ensemble.

| Standalone model/dead45 | spellF1 | Delta vs original short-TCN | Paired 95% CI of delta | Wins/7 |
|---|---:|---:|---|---:|
| Original short-loss TCN | 0.595755 | 0 | [0,0] | 0 |
| Spell-balanced TCN | 0.604291 | +0.008535 | [0.003403,0.012540] | 6 |
| Event-sampled TCN | 0.608023 | +0.012267 | [0.006049,0.017750] | 6 |
| Dual-scale TCN | 0.587528 | -0.008228 | [-0.014495,-0.003203] | 1 |

The spell-balanced loss and event sampling improve the standalone TCN, but neither replaces the control ensemble. Dual-scale drops even against the original standalone model; it is not adopted. The best ensemble uses the event TCN together with the old short model and SetSeq, rather than dropping both controls.

## Per match

| Match | Control | Primary event | Best event + consensus |
|---|---:|---:|---:|
| J03WMX | 0.546667 | 0.551154 | 0.551825 |
| J03WN1 | 0.566628 | 0.568349 | 0.569072 |
| J03WOH | 0.651744 | 0.655977 | 0.663372 |
| J03WOY | 0.641802 | 0.640699 | 0.640699 |
| J03WPY | 0.618994 | 0.617222 | 0.618333 |
| J03WQQ | 0.654496 | 0.665041 | 0.663243 |
| J03WR9 | 0.627871 | 0.634146 | 0.630395 |

## Short-possession diagnosis

This is an IoU-0.75 diagnostic, not the official primary metric, which remains the mean over IoU 0.3/0.4/0.5/0.6/0.7. All 3,061 GT H/A spells are kept. Count error is the mean per match; spell counts and matched counts are pooled.

| Method | Predicted/GT spells | Mean count error | Matched < 4 s / 1,621 | Matched 4 to < 15 s / 1,058 | Matched ≥ 15 s / 382 |
|---|---:|---:|---:|---:|---:|
| Control | 2208/3061 | -0.273891 | 230 | 539 | 301 |
| Primary event ensemble | 2235/3061 | -0.265543 | 239 | 539 | 300 |
| Best event + consensus | 2241/3061 | -0.263341 | 243 | 542 | 300 |
| Event standalone/dead45 | 2389/3061 | -0.214072 | 229 | 534 | 287 |
| Spell standalone/dead45 | 2426/3061 | -0.201483 | 215 | 540 | 285 |
| Dual standalone/dead45 | 2293/3061 | -0.245651 | 186 | 503 | 290 |

The best ensemble adds 33 predicted spells and 13 matched short spells over the control. [INFERENCE] Crop enrichment helps the short-possession information, but the event-standalone numbers show that raising the number of spells alone does not guarantee IoU matches. The best ensemble's gain is small and varies between matches.

The number of complete short events per training record, counted before sampling, is below. Each fold uses only the six matches other than the held-out one.

| Match | Eligible short events |
|---|---:|
| J03WMX | 278 |
| J03WN1 | 284 |
| J03WOH | 196 |
| J03WOY | 249 |
| J03WPY | 206 |
| J03WQQ | 178 |
| J03WR9 | 170 |

No match lacks events in this campaign; the behaviour of keeping a uniform window for a match without events is tested separately.

## Verification and artifacts

- 27 semantic tests pass: full-run duration/censoring, unknown gradients and the weighted denominator, view symmetry, event enrichment/source-match balance/half safety, original RNG/batch parity for non-event sampling, a dual-scale numerical/gradient proof and PP mask/permutation behaviour.
- 9 smoke seed fits of 1 update each on real training data; three identical full held-out inference replays. The smoke runs are not mixed into the 500-update performance estimates.
- 21 full training jobs, 63 seed fits, 31,500 actual updates; the metadata records six-match membership and every seed/parameter count. Every process exited 0.
- 21 saved-model ensembles were reloaded in separate processes; every full held-out probability array is identical to the cached outputs.
- Independent decoder rebuild: **140 identical arrays**, predecessor control byte-identical. This is an inference/decode reproduction, not a full training refit.
- The sum of job durations, **1,816.565 s**, covers loading, training, initial inference and saving; it excludes the separate replay subprocesses, registration, tests, final evaluation/rebuild and orchestration overhead. CUDA GPU was used, with GPU jobs run sequentially.

Manifest SHA-256: `2db5b8e793cfd43559425d600aa282422558e135016178944b3c8bfaf3d306ca`. The source and [PLAN.md](PLAN.md) were frozen after the smoke test and before the full fits, and were not updated to fit the results.

Aggregate exports: `results/spell_training_results.json`, `spell_training_protocol.json`, `spell_training_checks.json`, `spell_training_verification.json`, `spell_training_training.json`, `spell_training_comparisons.json`. Raw probability arrays/checkpoints/per-frame loss weights stay ignored in `out/`. No target CSV, new public/private evaluation. Component sources and reproduction commands: [README.md](README.md).
