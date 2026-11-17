# Wave5: possession boundaries, short spells and missing-ball robustness

> English version of the pre-registered plan. The original plan, whose SHA-256 was recorded before the runs, was written in Indonesian; it is kept in git history at commit `efdb82e`, path `experiments/exp_farras/wave5/PLAN.md` (`git show efdb82e:experiments/exp_farras/wave5/PLAN.md`). Because this file was replaced, the recorded PLAN hash no longer matches.

## Question

Wave4 improved public spellF1, but short possessions are still missed, the worst development count error is 44.62%, and Metrica 3 is worse than wave3. Do not conclude that the GRU mixture is needed just because its public score equals the PP-TCN's. This campaign tests separate training interventions, a team reset at dead balls, and rate calibration after all post-processing.

All wave1/2/3/4 code and inputs are read-only. Target GT is not used for training, ranking or choosing hyperparameters. The wave4 public score is already known; wave5 is adaptive research, not a new independent test.

## Data and split

- Training: the 19 licensed matches from wave4: 5 IDSSE development, 14 SkillCorner development. IDSSE uses the same 3 corruption rounds. No Metrica training is added.
- The six wave4 grouped folds are kept: matches sorted within each provider get index modulo 6, and all rounds of a match leave together. The six folds train on 15/15/16/16/16/17 matches. Normalisation and rate statistics come from the training fold only.
- Development: 23 scenarios / 21 matches: IDSSE 5 r0, SkillCorner 14 clean, Metrica 1/2 clean/degraded. Metrica is zero-shot stress and does not train the new models or the GRU prior.
- Objective: mean spellF1 per match, then the mean per provider, then the equal mean of IDSSE/SkillCorner/Metrica. MacroF1 tie-breaker, then family/decoder lexicographic. Do not drop short GT.
- Old holdout: IDSSE 2 r0, SkillCorner 6 clean, Metrica 3 clean/degraded. All have been evaluated before; do not call them blind.
- After the selection freeze, build stress pairs for the 6 SkillCorner holdout matches. Seed 20261113 plus a stable match identifier; drop an additional 35% of originally visible ball rows in contiguous spans of 10..60 samples, separately per half. Players, labels and timeline are unchanged. All X16 and PP/ball features are recomputed from the masked input. Stress is a new transformation test on old matches, not 6 new independent matches.

## Training three models

The wave4 PP-TCN base is unchanged: 64 channels, 6 residual Conv1d with kernel 5 / dilations 1,2,4,8,16,1; GroupNorm 8, dropout 0.15, crop 384, batch 16, 1,500 optimiser steps, AdamW lr 0.002 / weight_decay 0.01, OneCycle. Seeds 0/1/2, provider/match/round balanced sampling. Baseline mask_full 0.3, pooled-player noise 0.5; every PP channel that uses the ball is masked consistently. Full-block two-view inference; CPU checkpoint nets, actual CUDA fit after the smoke test passes.

1. `boundary_tcn`: add a boundary output head; the two state losses stay, plus 0.2 × boundary BCE. Positive where there is an adjacent change of known state, dilated ±2 frames within known labels, fixed positive weight 5. The first crop frame and changes adjacent to unknown are ignored.
2. `short_tcn`: the two state heads stay. Factor 2 on the state loss for live GT spells < 20 frames / 4 s. Duration is measured on the full record per half, not the crop; a run cut by unknown does not count as a complete short spell. The sampler is not changed at the same time.
3. `masked_tcn`: the two state heads stay; full ball-mask probability 0.6 instead of 0.3. Noise and the rest of the recipe unchanged.

Each model: six folds plus a final 19-match fit, 3-seed ensemble: 21 ensemble fits, 63 seed fits with the full recipe. The wave4 PP-TCN checkpoint is final only; the cached dev PP predictions are the real OOF. The final checkpoint is not used to fake held-out SkillCorner development predictions. Shipped Ryu weights are not used for IDSSE/SkillCorner OOF.

## Twelve probability families

`old_pp`, `old_gru25`, `boundary_tcn`, `short_tcn`, `masked_tcn`, `boundary_gru25`, `short_gru25`, `masked_gru25`, `quality_old`, `new_ensemble`, `new_ensemble_gru25`, `quality_new`.

- Old PP from the cached wave4 OOF on development, the final checkpoint zero-shot on holdout. GRU from the wave3 OOF IDSSE / zero-shot SC/Metrica prior already hashed by wave4. No anchor tree.
- `old_gru25` is exactly 75% wave4 PP + 25% old hybrid GRU. This is the frozen wave4 control.
- The three new `_gru25` families use 75% primary model + 25% GRU.
- `new_ensemble` is the mean joint probability of the three new models; the boundary is only the head actually trained in `boundary_tcn`, not an average with placeholders.
- Quality family: primary is old PP or the new ensemble; GRU weight `.05+.35*visibility*entropy(primary conditional-team)/log(2)`. Visibility is the centred 11-sample mean of X[:,3] within the half. Weight in [.05, .4]. A fixed rule; no calibration learned from GT or provider identity.
- Every mixture normalises the joint probabilities and recomputes the conditional dead/team. The primary boundary is kept; old PP without a learned boundary gets provenance false.

## Eight decoders

All without a dead-prior shift. Defaults: gate 0.35 m, kappa 0.05, tau 3, merge 1 s, dead threshold 0.5, min dead 2 s, rate_scale 1. Rate target from provider-balanced training spells per minute. The original wave4 lambda grid is used in full; nearest count, ties to the smallest lambda, no monotonicity assumption.

1. `wave4`: exactly the old `no_shift` decoder; rate calibration before merging, the intentional control.
2. `postmerge`: same whole-half H/A DP, but calibration counts the final labels after merging.
3. `live_reset`: postmerge plus an independent H/A DP in each live interval; team state is not carried across dead balls.
4. `boundary_reset`: live_reset with per-frame switch cost `lambda*(1-.5*boundary_score)`. Use the learned head when available; otherwise `clip(abs(diff(logit(p_team)))/4,0,1)` with a reset at the start of each half/live interval. Do not treat the 0.5 placeholder as a learned head.
5. `dead1`: live_reset, min dead 1 s.
6. `dead45`: live_reset, threshold 0.45, min dead 1 s.
7. `dead55`: live_reset, threshold 0.55, min dead 2 s.
8. `no_merge`: live_reset, merge 0.

12 × 8 = 96 configurations, 2,208 development evaluations. Count error and short/medium/long diagnostics are reported but do not replace the main metric.

## Selection and adoption decision before results

The frozen control is `old_gru25/wave4`. The full ranking uses the objective above. A new configuration is eligible to replace the control only if its mean spellF1 is at least 0.01 above the control, it wins on at least 2 of the 3 provider means, and no provider drops by more than 0.03. Pick the best eligible; if none, keep the control. This guards against small gains on development data that has been used repeatedly. Still record `best_new` by ranking outside the control combination so negative results stay transparent.

Before public: hash source/input/recipe/device, freeze the selection/ranking/checkpoint hashes. After that, only fixed candidates: `selected`, `best_new`, `previous_wave4`, `pure_boundary` (`boundary_tcn/wave4`). Identical roles are not independent models. The holdout has 16 scenarios including the 6 paired stress cases. Diagnostics per duration < 4 s, 4–15 s, ≥ 15 s use the official scorer's one-to-one IoU matching; short GT is not dropped from the denominator.

The independent refit retrains every new component needed by the selected candidate from the same seeds; frozen old components are only verified by checksum and by their earlier refit evidence. If the selected candidate uses no new component, do not claim a new-component refit. The 199,319-row CSV must be identical after refit/rebuild. The official public scorer is called only after this step, without changing the choice.

## Artifacts and limits

Code/reports in `experiments/exp_farras/wave5/` (now `experiments/E06_wave5/`), aggregates in the sibling `results/wave5_*.json` (now `experiments/E06_wave5/results/`). Raw data, prepared packs, features, probabilities, checkpoints and submission CSVs stay ignored. The source and artifacts of older waves are not changed. No automatic commit/push or tracking upload.

SkillCorner MIT and IDSSE CC BY 4.0, credit inherited from wave4/reference/SOURCE.md. Metrica only for evaluation. Statistics are treated as proxies and adaptive research; no claim of private/full-match performance, significance, or a fresh-match holdout that does not exist.
