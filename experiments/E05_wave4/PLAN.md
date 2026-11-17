# Wave4: TCN, multi-provider training, calibrated control

## Evidence before fitting

Pinned Ryu branch `5eaf3e595e91fa0502aac48037d0cae8c356ec6f`. Latest `experiments/017_multi_match/NOTES.md` reports public spellF1 about0.747, macroF1 about0.919 for the player-set TCN. Older `TECHNICAL_DOCUMENT.md` still says0.422 and is stale. `writeup.md` includes an intermediate0.6815. These are historical source-reported diagnostics, not interchangeable current checkpoint measurements.

Adopted source ideas:23 player-configuration features+5 ball channels in two team-oriented views; residual dilated TCN; per-player attention/max pooling; three seeds; random ball masking and input noise; independent dead/team heads; controlled-distance evidence; train-rate-calibrated two-state Viterbi; live sliver merging. Source ablations support player sets and multi-provider training, not blindly adding every rejected timing/touch/long-context idea.

Ryu shipped `weights_nometrica` trained27 matches:20 SkillCorner+7 IDSSE, excluding Metrica because its sample license is unclear. It **cannot** be a held-out model on those27. We therefore retrain each development fold from scratch and score shipped weights only on Metrica3/target as a diagnostic reference.

## Split and inputs

Licensed new-model training: IDSSE CC BY4.0 and SkillCorner MIT; no Metrica training. Reuse wave3's three corrupted rounds of IDSSE, original label-free25fps degrader and seeds. Acquire20 SkillCorner matches from public upstream URLs, verify tracking LFS SHA-256/size, retain metadata/phases provenance.

Development: first5 IDSSE J03WMX,J03WN1,J03WOH,J03WOY,J03WPY and first14 SkillCorner IDs listed in `prepare_skillcorner.py`. Locked campaign holdout: IDSSE J03WQQ,J03WR9 plus SkillCorner2010085,2011166,2013725,2015213,2016236,2017461. IDSSE holdout and Metrica3 were exposed in earlier campaigns; SkillCorner6 are new to Farras, but source Ryu research used them. No globally blind-study claim.

Six grouped development folds: sorted names independently within each provider, index modulo6. A held match and every corruption round are excluded from that fold's training. Final new models fit all19 development matches, then score Metrica1/2 clean and degraded zero-shot. Existing Metrica observations/labels remain diagnostic data, not new-model training data.

SkillCorner10fps tracking is converted with the source Ryu10-to25fps interpolation and containing-frame label rule, then sampled5fps per half. Labels come from phases of play (H/A in phase, D elsewhere inside period, unknown outside). Provider definitions and extrapolated tracking are not corrected challenge labels.

## Fixed model grid

Three architectures: `plain_tcn` (23player+valid+5ball), `pp_tcn` (same plus10-feature per-player attention/max), `control_tcn` (PP plus7 scaled Farras controls). Width64, six residual Conv1d blocks, kernel5, dilations1,2,4,8,16,1, GroupNorm8, dropout0.15. Player MLP width48. Two binary heads dead/team; no GRU or change-point head.

Each fit trains3 seeds0/1/2,1500 steps each, batch16,crop384, AdamW lr0.002,weight_decay0.01,OneCycle. Ball full-crop masking0.3 plus source span mask; input noise0.5 on standardized23 player channels. When hidden, all ball-dependent PP/control channels are masked too. Sampling balances provider, then match, then corruption round; no weighting by frame labels or test outcomes.

Features are adapted to actual5fps: physical velocities,2-second rolling player stats and1-second ball-speed median, reset per half. Nearest-opponent distance is computed relative to each player's actual opposing team; the original source view-dependent shortcut gives zero distance to self for some opposite-view players. This correction and5fps feature construction mean new models are **adaptations**, not byte-identical Ryu models. Unknown sampled target-team slots remain absent rather than guessed.

Prefer CUDA after actual convolution training/refit/process-exit smoke. Previous cuDNN GRU failure does not establish Conv1d failure. If a backend fails, choose a verified explicit backend before full fitting and record it; do not suppress process failure or reduce model/steps.

Pretraining runtime decision: CUDA Conv1d+PP smoke passed all3 architectures, each with3 seeds, independent refit and checkpoint roundtrip, with exit0. All11 behavioral regressions passed. Use `FARRAS_TRAIN_DEVICE=cuda` for new TCNs; prior wave3 GRU inference stays in its separate CPU process. Shipped3-seed Ryu checkpoint architecture was loaded successfully without inspecting holdout outcomes.

## Probability families and decoders

Eight probability families:3 new TCN ensembles; PP-TCN mixed with prior wave3 hybrid GRU at25/50/75% other weight; PP-TCN mixed25% prior tree; prior wave3 probability combination. Prior predictions use exact earlier out-of-fold predictions on IDSSE, zero-shot final models on SkillCorner, and excluded-Metrica anchor folds for Metrica1/2. No leaked in-sample new-model folds.

Eight decoder settings, each otherwise matching source Ryu: `ryu_rate` gate0.35m,kappa0.05,merge1s; `gate10` gate1m; `merge05` merge0.5s; `kappa02` kappa0.02; `no_shift` no dead-prior shift; `fixed2` lambda2 instead of rate calibration; `rate08` train-rate scale0.8; `rate12` scale1.2. Dead threshold0.5,min duration2s, target dead prior0.407 from external LaLiga effective-time source, tau3. Team evidence persists across D as in Ryu, then D overlaid; sliver merging only inside live runs. Half blocks reset.

Calibration uses only train-source-balanced mean possession rates/dead shares. Evaluate the source lambda grid0.05..120 and choose the grid value nearest target count, tie smallerlambda; this avoids assuming monotonic counts. Counts calibrated before merging, never against test labels. Duration in seconds and5fps scale explicit. This grid-nearest choice differs from original log interpolation and is disclosed.

8 probability families x8 decoders =64 configurations, no grid extension after results.

## Selection, freeze, verification

Primary objective: equal mean of provider-specific mean spellF1 for IDSSE5 held matches, SkillCorner14 held matches, and Metrica2 zero-shot matches (clean/degraded averaged inside each Metrica match). Ties macroF1,family,decoder. Report count error as warning, do not silently replace competition metric or remove short true spells for selection.

After freeze evaluate chosen candidate, new PP-TCN+ryu_rate, previous frozen wave3 combination on10 holdout scenarios (2IDSSE+6SC+Metrica3clean/degraded). Shipped Ryu reference only on Metrica3clean/degraded, never advertise in-sample IDSSE/SC predictions as held-out. Then public sanity for four fixed candidates only. No64-way public sweep or selection changes afterward.

Independent selected-model refit and exact199319-row target CSV/hash reproduction required. If selected family reuses prior wave3 models, retain that campaign's independent-refit proof and verify unchanged checkpoint hashes; refit every newly trained component. No automatic commit/push.

## Execution and artifacts

Same format as prior campaign: separate `wave4/`, parent owns plan/harness/freeze and reports; six workers implement data/features/models/decoder/fusion/reference only. Commands `prepare_skillcorner.py`, `prepare_features.py`, `prepare_prior.py dev`, `run.py dev`, `prepare_prior.py holdout`, `run.py holdout`, `run.py public`. Cache, source weights, raw tracking and frame-level CSV remain local/ignored; share only aggregate metrics and source attribution.
