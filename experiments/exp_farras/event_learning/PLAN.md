# Event learning: fixed experiment protocol

## Starting evidence and controls

Previous spell_training produced an event-sampled standalone TCN with mean official spellF1 IDSSE0.6080225797645553, compared with original short-loss TCN0.5957554001098855. Its fixed short50% + SetSeq25% + event25% ensemble reached0.6189411362728311/dead45 and0.619562844991005/consensus. The primary gain against legacy ensemble was below the preregistered0.005 threshold, so it was not promoted. Spell-balanced standalone loss also improved; shared-kernel dual-scale architecture lost.

This campaign retains the useful event sampler and tests six training extensions rather than another decoder scan. Strong research control is the preceding exploratory winner: short50% + SetSeq25% + previous event25%, frozen consensus(alpha0.5/contact_beta0.5), mean0.619562844991005. This previously exposed winner is a new comparison baseline, not a retroactive deployment promotion. Legacy short75% + SetSeq25%/dead45 remains reported separately at0.6154575861548992.

Parent fixes planning, recipes, interfaces, integration, execution and selection. Independent workers implement only assigned objectives, averaging, batch preparation, provenance validation and protocol review. No source changes in teammates' folders or old frozen campaigns.

## Shared training and evaluation

- Seven original IDSSE matches, seven LOSO folds, exactly six other unique matches per fit. No GT orientation, oracle D/X masking or heldout label calibration.
- Same r0 packs, PPfeature packs, provider labels, unknown handling, half boundaries and5fps-to25fps block expansion as prior benchmarks. No data download or new corruption regime.
- All new families use50% event-centered crop replacement after the existing balanced source/match/variant/window/view sampler. Complete H/A spells shorter than20samples, strict half/unknown censoring, jitter[-96,96], half-clamped384sample crops. No global oversampling of event-rich matches.
- Width64, six Conv5 residual blocks, dilations(1,2,4,8,16,1), PPencoder48x10, GroupNorm8, GELU, dropout0.15. Three seeds0/1/2,500optimizer updates per seed,batch16,crop384,AdamWlr0.002/weight_decay0.01,OneCycleLR, original mask_full0.3/noise0.5. Fresh random initialization, no predecessor weights loaded.
- Six families x7folds x3seeds =126newseedfits/63,000updates. Same updates and sampler, not universal parameter/input parity: auxiliary boundary adds one output channel, contact adds seven observation channels, EMA keeps a training shadow copy. Counts and runtime reported, not hidden.
- Inference uses complete halves, two views and three-seed averaging. The boundary head is auxiliary only; all decoders see joint H/A/D probabilities derived from the original first two heads. No learned boundary decoder added.
- Main metric official mean H/A spellF1 across IoU0.3/0.4/0.5/0.6/0.7, every short spell retained. MacroF1 secondary. IoU0.75duration recall diagnostic only.

## Six fixed recipes

### balanced

Combine event sampling with preceding complete-spell team loss. Full-match H/A weights `clip(sqrt(20/L),0.25,4)`, censored H/A weight1, D1, unknown0; compute before cropping. Ordinary known-frame dead BCE plus weighted live-team BCE divided by weighted live mass. Reuse frozen spell_training weighting implementation. This tests combination of two individually useful training changes.

### focal

Keep original short-weighted dead BCE/known count. Replace team BCE with `short_weight * (1-p_t)^2 * BCE`, divided by unweighted live count, with `p_t=exp(-BCE)` for numerical stability. Gamma2, no class alpha or additional normalization. This intentionally changes hard-example emphasis and relative team-loss scale; no hyperparameter scan.

### boundary

Keep original short-loss state objective. Replace two-channel head with three channels and add0.2times boundary BCE, positive weight5, divided by valid boundary-frame count. Targets follow the existing radius2 boundary-target definition, computed on each full training half before crops. Unknown intervals cannot propagate targets; half starts invalid. Cropping a full target does not invent a new boundary at crop start. Auxiliary channel is ignored at inference.

### ranking

Keep original short-loss objective. At adjacent training frames whose labels are different and both H/A, add0.1times mean `softplus(1-sign*(z[t]-z[t-1]))`, where z is conditional-team logit and sign is positive when current team equals sampled view, negative otherwise. Skip D/unknown pairs; no penalty on unchanged pairs. Each crop remains within one half. No eligible changes gives a differentiable zero auxiliary term.

### ema

Keep preceding event TCN training objective and sampler unchanged. Make an initial detached shadow copy, then after every optimizer update perform parameter EMA with decay0.99. Copy buffers directly, including integer buffers. Persist the500-update shadow model for each seed, not a validation-selected checkpoint. No heldout model selection, prediction averaging over checkpoints, initialization debiasing or extra optimization steps. Compare initialization/RNG and model-serialization semantics in tests.

### contact

Keep original event sampler and short-loss objective, but attach all seven audited raw25fps contact/memory/dead-cue controls from cross_branch. Both training views use their correct signs/team-distance swaps; no GT lookup. Original control_tcn architecture accepts36channels versus29. Apply the exact ball dropout mask to these seven channels as well. Same PPencoder, temporal network and heads. This combines better sampling with previously weaker contact/context features rather than claiming another raw-input-parity experiment.

## Fixed probability families and decision

Two old families: legacy probability and event research-control probability. For each new model N:

- pure: N;
- replace: `.50*short + .25*SetSeq + .25*N`;
- split: `.50*short + .25*SetSeq + .125*previous_event + .125*N`.

Twenty probability families, each with frozen dead45 and consensus(alpha0.5/contact_beta0.5), yield40settings x7matches=280metric rows. Consensus directional agreement always uses the original heldout short/SetSeq predictions and observation-only controls, never new training labels. No post-score tuning of decoder, weights or settings.

Primary `balanced_replace/consensus`, compared against **event_control/consensus0.619562844991005**. Promotion requires gain>=0.005, wins>=5/7matches, and paired match-bootstrap95%lower>0;5,000draws, seed20261118. If primary fails, retain research control; report raw best as exploratory without a best-fallback substitution. Legacy comparison is secondary and cannot establish promotion against the stronger baseline.

All seven matches, prior best and hypotheses are previously exposed. This is adaptive research, not a fresh blind estimate. Confidence intervals are nominal, without multiple-comparison or adaptive-history correction. No supervised meta-fit on OOF predictions from other outer folds, because those base models could have seen the outer test match.

## Verification, freezing and publication

Before full fits: semantic tests for mathematical losses/gradients/censoring, event-batch parity and contact/auxiliary alignment, EMA mutation/buffer/serialization behavior, fold/seed/budget/probability validators. Then18real smoke seedfits (six recipes x3seeds,1update) trained on six real matches, fullheldout prediction, and six isolated saved-model inference replays. Smoke is not a performance estimate.

After checks and before any new500-update fits, hash code, this plan, input features/probabilities and predecessor manifests/results/proofs. Each of42model-family/fold jobs uses an isolated process; GPU work sequential. Persist fresh weights, normalized probabilities and actual training membership/update/parameter audits. Another isolated process reloads each saved ensemble and requires exact fullheldout probability equality. Independent final rebuild reproduces280decoded arrays and verifies all42saved-model replay proofs and both old controls byte-for-byte.

Raw caches, labels, predictions and weights remain under ignored out/. Only code/docs and aggregate JSON reports are shareable under experiments/exp_farras. No new public/private or target evaluation, no target CSV, no automatic commit or push. Previous frozen plans and hash lists are not patched to hide errors; any protocol change after registration requires a separate campaign.
