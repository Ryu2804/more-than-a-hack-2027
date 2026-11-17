# Event learning results

## Decision

New best exploratory **`contact_replace/consensus`: mean official IDSSE spellF1 0.6245765534033564**, macroF1 0.9115530617305891. The strong comparison `event_control/consensus` stays at 0.619562844991005. Gain **0.005013708412351555**, wins 5/7, paired match-bootstrap 95% CI **[−0.0002117957432191992, 0.011265809452937545]**. The gain and the win count pass two of the thresholds, but the interval still crosses 0. It is also not the primary fixed before the run, so it is not promoted automatically.

Primary **`balanced_replace/consensus` reaches 0.6191920412910819**: delta −0.00037080369992297565, CI [−0.0033139818714818287, 0.002354190193876774], wins 3/7 with 1 tie. The primary fails; the strong research control is kept. The decision thresholds were not changed after seeing the results.

The exploratory alternative `ranking_replace/dead45` reaches 0.6215116612127324 and wins **7/7 against the strong consensus control**, delta 0.0019488162217274738, CI [0.0007000594338442669, 0.003537213935308925]. The gain is below 0.005 and it is not the primary. Compared with the event control using the same dead45 decoder: delta 0.0025705249399012547, CI [−0.0003631786146264327, 0.005429331168577739], wins 5/7. So the 7/7 result is a full-pipeline comparison, not evidence of an isolated ranking-loss effect with an identical decoder.

The best versus the legacy ensemble 0.6154575861548992: gain 0.00911896724845713, CI [0.0023922816245338916, 0.01569762203059722], wins 6/7. This easier comparison is not used in place of the requirement against the strong control.

All intervals use 5,000 paired bootstrap draws, seed 20261118, unit = unique match. The seven matches, hypotheses and earlier controls had been exposed; the nominal intervals do not correct for the adaptive history or multiple comparisons. This is IDSSE research, not a fresh blind test or a public/private competition score. No target evaluation or new submission.

## The six recipes that were run

All use the 50% event sampler, three seeds 0/1/2, 500 updates/seed, batch 16, crop 384, seven LOSO folds and the other six training matches. Total **126 new seed fits / 63,000 updates**. No training on the held-out match and no predecessor weights loaded. Short/SetSeq/previous-event probabilities are used only from authenticated OOF caches.

| Recipe | Change from the previous event TCN | Parameters/network |
|---|---|---:|
| balanced | Complete-spell inverse-sqrt team loss, ordinary dead BCE | 135219 |
| focal | Conditional-team focal loss, gamma 2, original dead loss | 135219 |
| boundary | Auxiliary radius-2 boundary target/BCE, third head | 135284 |
| ranking | Signed adjacent H/A-change margin loss | 135219 |
| ema | EMA 0.99 after every optimiser update; shadow used for inference | 135219 |
| contact | Seven observed raw 25-fps contact/memory/dead-cue channels | 135667 |

The boundary target is computed on the full training half before cropping and does not cross unknown labels or halves. Only the first frame of each half is invalid, not the whole half. The auxiliary head is not used by the decoder. EMA does not pick a checkpoint based on validation. Contact uses the already-audited cached two-view features, without GT orientation/D/X mask; all seven channels follow the ball-dropout mask. The complete-spell balanced weights and the original short weights have different censoring rules, deliberately kept.

The update budgets are equal, but inputs/parameters/compute are not identical across recipes. Boundary changes the head, contact adds inputs and EMA keeps a shadow copy during training. The formulas and decisions are frozen in [PLAN.md](PLAN.md).

## All 40 configurations

`replace`: short 50% + SetSeq 25% + new model 25%. `split`: short 50% + SetSeq 25% + previous event 12.5% + new model 12.5%. `pure`: the new model alone. Two fixed decoders: dead45 and consensus (alpha 0.5 / contact_beta 0.5), without retuning. Deltas and wins are against the strong research control, not the legacy one.

| Configuration | spellF1 | macroF1 | Delta vs strong control | Wins/7 |
|---|---:|---:|---:|---:|
| `contact_replace/consensus` | 0.624577 | 0.911553 | +0.005014 | 5 |
| `contact_replace/dead45` | 0.623893 | 0.911369 | +0.004330 | 4 |
| `ranking_replace/consensus` | 0.621600 | 0.910293 | +0.002037 | 4 |
| `ranking_replace/dead45` | 0.621512 | 0.910192 | +0.001949 | 7 |
| `contact_split/consensus` | 0.620972 | 0.910710 | +0.001409 | 4 |
| `boundary_split/consensus` | 0.620675 | 0.910270 | +0.001112 | 4 |
| `ranking_split/consensus` | 0.620519 | 0.910168 | +0.000956 | 6 |
| `contact_split/dead45` | 0.620461 | 0.910488 | +0.000898 | 3 |
| `balanced_split/consensus` | 0.620023 | 0.910201 | +0.000460 | 5 |
| `boundary_replace/consensus` | 0.620014 | 0.910291 | +0.000451 | 3 |
| `boundary_split/dead45` | 0.620003 | 0.910203 | +0.000440 | 4 |
| `boundary_replace/dead45` | 0.619767 | 0.910271 | +0.000204 | 4 |
| `event_control/consensus` | 0.619563 | 0.909983 | +0.000000 | 0 |
| `ranking_split/dead45` | 0.619542 | 0.910006 | -0.000021 | 3 |
| `contact_pure/consensus` | 0.619249 | 0.907418 | -0.000313 | 4 |
| `balanced_split/dead45` | 0.619238 | 0.910038 | -0.000325 | 4 |
| `balanced_replace/consensus` | 0.619192 | 0.910290 | -0.000371 | 3 |
| `event_control/dead45` | 0.618941 | 0.909976 | -0.000622 | 3 |
| `ema_replace/consensus` | 0.618872 | 0.909735 | -0.000691 | 3 |
| `ema_split/consensus` | 0.618762 | 0.909831 | -0.000801 | 2 |
| `ema_split/dead45` | 0.618524 | 0.909732 | -0.001039 | 2 |
| `balanced_replace/dead45` | 0.618348 | 0.910172 | -0.001215 | 2 |
| `ema_replace/dead45` | 0.618150 | 0.909687 | -0.001413 | 2 |
| `contact_pure/dead45` | 0.617985 | 0.907221 | -0.001578 | 4 |
| `focal_split/consensus` | 0.617877 | 0.910107 | -0.001686 | 4 |
| `legacy/consensus` | 0.616922 | 0.909863 | -0.002641 | 3 |
| `focal_split/dead45` | 0.616871 | 0.910046 | -0.002691 | 3 |
| `legacy/dead45` | 0.615458 | 0.909587 | -0.004105 | 2 |
| `balanced_pure/consensus` | 0.615189 | 0.903583 | -0.004373 | 2 |
| `balanced_pure/dead45` | 0.614202 | 0.903348 | -0.005361 | 2 |
| `focal_replace/consensus` | 0.613417 | 0.909840 | -0.006146 | 3 |
| `focal_replace/dead45` | 0.612647 | 0.909697 | -0.006916 | 2 |
| `ranking_pure/consensus` | 0.611603 | 0.904330 | -0.007960 | 2 |
| `ranking_pure/dead45` | 0.610870 | 0.904315 | -0.008693 | 1 |
| `boundary_pure/consensus` | 0.608296 | 0.904209 | -0.011267 | 1 |
| `boundary_pure/dead45` | 0.608225 | 0.904188 | -0.011338 | 1 |
| `ema_pure/consensus` | 0.605611 | 0.903409 | -0.013951 | 1 |
| `ema_pure/dead45` | 0.605592 | 0.903368 | -0.013971 | 0 |
| `focal_pure/consensus` | 0.582393 | 0.901827 | -0.037170 | 0 |
| `focal_pure/dead45` | 0.580623 | 0.901463 | -0.038940 | 0 |

The primary metric remains the official mean spellF1 across IoU thresholds 0.3/0.4/0.5/0.6/0.7, over all H/A spells including short ones. MacroF1 is secondary. All 280 configuration-match rows use the same labels, unknown masking and 25-fps expansion.

## Secondary standalone contrasts

Exploratory comparisons after the run; they do not change the primary decision. All use dead45 and compare with the previous event standalone model at 0.6080225797645553, not the control ensemble. Parameter/input parity differs for boundary/contact as described above. This judges the training recipes more directly than comparing pure models with a strong ensemble.

| Standalone recipe | Delta vs previous event | Paired 95% CI of delta | Wins/7 |
|---|---:|---|---:|
| balanced | +0.006179 | [-0.001047,0.014312] | 5 |
| focal | -0.027400 | [-0.041041,-0.013547] | 0 |
| boundary | +0.000203 | [-0.005157,0.005124] | 3 |
| ranking | +0.002848 | [-0.000978,0.007167] | 5 |
| ema | -0.002430 | [-0.006575,0.001624] | 2 |
| contact | +0.009963 | [-0.006913,0.026013] | 5 |

Contact gives the highest mean, but the variation between matches is still large. Focal drops on every match in the standalone contrast; EMA 0.99 does not beat the previous event model on the mean. The auxiliary boundary is almost neutral. Balanced raises the standalone point estimate but does not automatically improve the primary ensemble. [INFERENCE] Standalone gains and complementarity with short/SetSeq are different things; stacking two recipes that once improved does not guarantee a better ensemble.

## Per match

| Match | Strong control | Primary balanced | Best contact | Ranking/dead45 |
|---|---:|---:|---:|---:|
| J03WMX | 0.551825 | 0.554070 | 0.557959 | 0.555152 |
| J03WN1 | 0.569072 | 0.574570 | 0.590607 | 0.575199 |
| J03WOH | 0.663372 | 0.659302 | 0.664723 | 0.664151 |
| J03WOY | 0.640699 | 0.633540 | 0.635705 | 0.641884 |
| J03WPY | 0.618333 | 0.616874 | 0.625313 | 0.619945 |
| J03WQQ | 0.663243 | 0.665591 | 0.668299 | 0.663599 |
| J03WR9 | 0.630395 | 0.630395 | 0.629429 | 0.630653 |

## Short-possession diagnostic

IoU 0.75 here is a diagnostic only, not the primary. All 3,061 GT H/A spells are kept. Predicted/matched spell counts are pooled; count error is the mean per match.

| Pipeline | Predicted/GT spells | Mean count error | Matched < 4 s / 1,621 | Matched 4 to < 15 s / 1,058 | Matched ≥ 15 s / 382 |
|---|---:|---:|---:|---:|---:|
| Strong event control | 2241/3061 | -0.263341 | 243 | 542 | 300 |
| Primary balanced | 2253/3061 | -0.259366 | 248 | 550 | 300 |
| Best contact | 2242/3061 | -0.262804 | 263 | 546 | 304 |
| Ranking replace/dead45 | 2259/3061 | -0.257845 | 247 | 547 | 300 |

The best contact pipeline adds 20 matched short spells while the number of predictions grows by only 1. [INFERENCE] This signal fits better team assignment/timing rather than simply forcing more spells. It does not yet prove generalisation to new matches; the difference between the best and the strong control still has an interval crossing 0.

## Verification and publication boundary

- 77 semantic tests pass: numerical/gradient loss references, unknown masks, full-half boundary/crop semantics, contact batch/view/mask parity, EMA mutation/buffers/serialisation, and exact rejection of fold/seed/budget/probability/replay mismatches.
- 18 real smoke seed fits of 1 update each, six identical full held-out checkpoint replays. The smoke runs are not mixed into the 500-update performance estimates.
- Before the full fits, 28 predecessor control decoded arrays were reproduced byte-identical: two probability controls, two decoders, seven matches.
- 42 full model-family/fold jobs completed, 126 actual seed fits / 63,000 updates. Actual training names, seed order, update counts, EMA update counts and per-family parameter counts were validated.
- 42 saved-model ensembles were reloaded in separate processes and produced identical full held-out probability arrays. Every fit/replay/evaluate/rebuild process exited 0.
- Independent rebuild in another process: **280 identical decoded arrays**, predecessor controls still byte-identical. This is an inference/decode reproduction, not a second full training refit.
- Sum of the 42 job timers after manifest verification: **3,685.890 s**, covering data loading, fitting, initial inference and saving. It excludes startup/source verification, separate replay processes, semantic/smoke checks, final evaluate/rebuild and orchestration overhead. CUDA GPU, sequential.

Manifest SHA-256: `14327721dbf19145c288c54fd95f78564ec427b64d284f070ba1788f8d6095e1`. Source/PLAN/input/provenance were frozen before the full fits and not revised to fit the new scores.

Aggregate exports: `results/event_learning_results.json`, `event_learning_protocol.json`, `event_learning_checks.json`, `event_learning_verification.json`, `event_learning_training.json`, `event_learning_comparisons.json`. Raw tracking, labels, weights, probabilities and per-frame targets stay ignored in `out/`. No new public/private score, target CSV. Reproduction commands and component sources: [README.md](README.md).
