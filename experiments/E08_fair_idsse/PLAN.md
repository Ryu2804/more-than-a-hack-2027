# Fair IDSSE benchmark: protocol v1

> English version of the pre-registered plan. The original plan, whose SHA-256 was recorded before the runs, was written in Indonesian; it is kept in git history at commit `efdb82e`, path `experiments/exp_farras/fair_idsse/PLAN.md` (`git show efdb82e:experiments/exp_farras/fair_idsse/PLAN.md`). Because this file was replaced, the recorded PLAN hash no longer matches.

Goal: replace the consolidation ranking that mixed adapters, training membership and GT exposure. This is a benchmark of model families adapted to the same data/protocol, not an exact reproduction of the team's submissions or a final branch ranking.

## Data, split, labels

Seven IDSSE matches J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9. Use the same prepared wave3 r0 data, byte for byte, for every arm; corruption seed 20261108 + match index. No r1/r2, SkillCorner, Metrica, BAR-ATH target, public GT or old checkpoints. Team mapping from the match metadata home/away, not from agreement with GT. The degrader receives tracking/team/half/seed, not labels. Original provider H/A/D labels, unknown −1, no merging/filtering of short GT.

Leave-one-match-out, 7 folds. Every trained arm must use exactly the other 6 matches, not 2 matches or final weights that have seen the test match. Unknown training labels are ignored; unknown test labels are used only by the scorer, not at inference. Normalisation/rate/dead_share come only from the 6 training matches. Half-local 5 fps is mapped to 25 fps using the same blocks25 for every output. The runner checks the timeline and prediction size; it gives the predictor no hidden labels, X mask or GT orientation.

Every match has been exposed in team research. This LOSO gives new match-exclusive fits, not a fresh blind test or evidence of private superiority.

## Arms and fixed budget

1. HGB symmetric: Farras's wave2 family, all 6 matches in the training pool; deterministic balanced sample of up to 40,000 rows after H/A augmentation, 100 iterations, 15 leaves, lr .08, l2 = 1. No truncation to the first 2 matches. The X16 features have the same origin as for every model.
2. SetSeq: Jeremy's DeepSets+BiGRU family, the adapted wave3 setseq implementation (not exact exp7). Width/hidden 32, state CE + boundary BCE 0.5, pos_weight 3, mirror augmentation, AdamW lr .002 wd .0001, clip 1.
3. PP-TCN: Ryu's family adapted in wave4, width 64, dilations 1/2/4/8/16/1, mask_full .3, noise .5, AdamW lr .002 wd .01, OneCycle.
4. Short-loss PP-TCN: Farras's wave5 family; two heads, loss factor 2 on complete live spells < 4 s; the rest of the recipe as PP-TCN.
5. Touch-hold: Rasya's family adapted to 5 fps, radius .8 m, minimum carried run 1.2 s; dead cue: out of pitch ≥ .6 s or nearest-player speed < 2 m/s for ≥ 1 s, visible ball/valid metadata required. Speed from position in metres / elapsed seconds, not / 3.6. Half reset, leading pre-touch D, fixed dead merge 0. No GT flip, GT dead mask or GT X mask. This is a label-free rule, different from the original oracle-dependent implementation.

All neural arms: 3 seeds 0/1/2, 500 optimiser updates/seed, batch 16, window/crop 384. SetSeq uses explicit fixed-step training, not a hidden epoch budget. Total 63 fresh seed fits / 31,500 updates; HGB 7 fits, rule 0. The budget was fixed before the results were known and is smaller than the 1,500-step production recipe; the interpretation is an equal-update-budget benchmark. Equal steps are not equal FLOPs/wall time. Optimiser/loss/augmentation and model representation still differ and are reported; no claim of a causal architecture-only ablation. No tuning or epoch selection based on the outer fold.

## Decoder and metric

Every learned joint probability gets the identical frozen wave5 dead45 decoder: threshold .45, min D 1 s, live-reset H/A DP, gate .35 m, kappa .05, tau 3, merge 1 s, full 18-lambda grid with the nearest fold-training rate after merging. Learned boundary heads are not used by the decoder, including SetSeq's. The argmax of each learned arm is reported separately as the same diagnostic. The rule's hard labels go in a native-rule panel; there is no claim that it gets the same decoder. Avoid comparing branch-specific tuned decoders in the main table.

Official metric: mean spellF1 at IoU .3/.4/.5/.6/.7, all short GT; macroF1 secondary. Unweighted mean over 7 unique held-out matches. No submission-candidate selection or promotion based on the table. Paired bootstrap 5,000 (seed 20261114) over matches, not frames/seeds; 95% percentile, exploratory, not familywise corrected. All method pairs are reported so the winner is not singled out.

## Integrity, verification, limits

PLAN/source/dependency files, the IDSSE provenance audit, data packs and feature packs are hashed before the fits. Each job cache is bound to the manifest hash, exact training names, held-out name, family, seeds and steps; a mismatch fails instead of being reused. The predict API rejects labels and overlapping training/test names. Old weights are never opened. No fallback that patches old results.

Correction of an earlier audit: Ryu's reference file does use a view-relative opponent mask, but our actual feature builder `wave4/features.py:_player_set` uses `player_team[j] != player_team[k]`, not the view mask. Assuming the bug in the reference copy is automatically inherited by our model is wrong. The benchmark uses the same actual-builder feature packs for both PP arms, not a copy of the reference builder. The default decoder was chosen earlier on exposed data, although it is not tuned here. All arms are offline full-half, not streaming.

Before the freeze, run the semantic tests, an actual reduced 1-step fit/predict of each neural arm on a real data slice, HGB fit/predict and the actual rule. The full benchmark must finish every fold; the report must not mix partial/old scores. An independent fresh-process rebuild of decoding from the output caches for every fold/arm is verified by label hashes. It is not called a model refit.

Raw tracking, per-frame labels/probabilities, checkpoints and out/cache are ignored. Only code/protocol and aggregate metrics/provenance are shared. No commit/push; the team's source and wave1..6 are immutable. IDSSE CC BY 4.0; provenance/source credits inherited from wave3/reference/SOURCE.md and wave4/reference/SOURCE.md.

## Runtime correction before full pre-registration

32 semantic tests pass. The combined smoke test completed every fit/predict and wrote the JSON, but the Windows process exited fatally with code −1073740791. That result is not accepted as a runtime pass. Before the full run, each family/fold is moved to its own child process, and the parent only scores. A child must exit 0 before its cache is read; errors are not swallowed. Budget/data/model unchanged. A new smoke test verifies each process separately before the full-fit registry is created.

Sampling is also part of the recipe: SetSeq takes windows in proportion to the number of valid starts, the PP-TCN balances matches, HGB takes a balanced sample from the global pool. All of them may access the same 6-match pool, but the sampling/loss distributions are not forced to be identical. The results compare fixed adapted pipelines under a common evaluation, not a causal experiment that swaps only the architecture.

The SetSeq-only CUDA child also finished computing but failed to exit, with status 3221226505; isolation alone was not enough. Before the freeze, the backend was changed: SetSeq on CPU, both PP-TCNs on CUDA; HGB/rule on CPU. Updates/seeds/data unchanged. The registry binds the backend per family and the worker verifies it. Because the hardware differs, the results are not used for latency/efficiency ranking. Acceptance still requires exit 0, not just the presence of a JSON.
