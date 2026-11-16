# Provider generalization: sealed research protocol

## Public-window and exposure audit

The competition public window contains17H/A possessions. RULES.md73-84 and EVALUATION.md125-142 permit pipeline-correctness checks, not training, parameter tuning or choosing between versions from that window. Older wave3/4/5/6 runners selected from external development, froze the choice, and subsequently reported public diagnostics. That code order is evidence against direct within-campaign public-score selection, not proof that previously observed public outcomes never influenced later research. The recent IDSSE-only campaigns did not evaluate public GT again.

All seven IDSSE and twenty SkillCorner matches available locally have prior research exposure. This is documented in earlier plans/results; prior shipped weights also used those cohorts. The current SkillCorner match tree `0c809ab9eabe28ebcd0b28a6916617cde88f852c`, [source inventory](https://api.github.com/repos/SkillCorner/opendata/git/trees/0c809ab9eabe28ebcd0b28a6916617cde88f852c), contains the same20matchIDs. No genuinely new blind cohort has been established. Do not claim no overfitting, an unbiased fresh test or private-match improvement.

Parent owns planning, integrations, orchestration, training and decisions. Six independent slices implement fixed regularizers, read isolation, prediction sealing and evidence audit. Old frozen campaigns and teammates' experiments remain immutable.

## New evaluation separation

Two source-only transfer directions, fresh weights:

1. Train on all7IDSSE matches, r0only. Test on all20SkillCorner matches, original clean preparation.
2. Train on all20SkillCorner matches, cleanonly. Test on all7IDSSE matches, r0only.

No match/provider appears in both source and target of a fit. This deliberately tests provider shift, not the same-provider LOSO metric from earlier campaigns. Do not compare scores directly to0.624577IDSSE ensemble as a better/worse leaderboard ranking. Do not reuse shipped weights, old OOF predictions, SetSeq mixtures or source-target training augmentation.

Normalization, short-spell weights, event indices, possession-rate/dead-share statistics and all optimization use only source-provider training matches. Explicit original loader calls use r0for IDSSE; inherited multi-round training helper is not used. Target inference consumes dedicated observation-only carriers: no y/y25members. Test labels are copied once into a separate external-label vault during preparation, not passed into fit, model inference or decoder. No target/competition data or GTis prepared or scored.

Trusted-Python runtime read audit hooks deny competition evaluation/gt, target packs, competition assets, previous outputs/results and role-specific label-bearing inputs. Every pipeline process intentionally probes denied competition-public access and records the expected denial without reading bytes. Fit additionally denies the target provider's original labeled packs and feature packs and the entire label vault. Predict/replay deny every original label-bearing pack and the vault. This is a code-level guard, not an OSsandbox or independent security certification. Any unintentional blocked read aborts, not a fallback.

## Five fixed training recipes

Base event TCN: original PPencoder48x10,29global channels,width64,six Conv5residual blocks with dilations(1,2,4,8,16,1),GroupNorm8,GELU,dropout0.15,two output heads.135.219parameters. Training source/match/variant/window/view sampling followed by50%event-centered replacement on complete H/Aspells<20samples, jitter[-96,96],384sample windows confined to one half. Original short-loss weights2for H/Aspells<20samples, same original censoring and unweighted known/live denominators. Training weights calculated before crops.

All recipes: seeds0/1/2,500optimizer updates/seed,batch16,crop384,AdamWlr0.002/weight_decay0.01,OneCycleLR, original ball mask_full0.3 and pooled-player noise0.5. Separate augmentation RNG from window/batch RNG, so new regularizers do not change the baseline crop sequence. Three-seed/two-view full-half inference with no augmentation. Ten ensemble fits,30seedfits and15.000optimizer updates. RDrop uses two forward/backward graphs per update; equal parameter/update budgets do not imply equal compute.

- **control**: source-only event TCN, no additional regularizer.
- **pp_dropout**: during training, independently drop20%of PPslots per example, with the same keep/drop identity across the entire crop. Modify PPmask only; token values, pooled features and ball stay unchanged. Empty PPbranches allowed; no forced keep. This is representation regularization, not re-simulation of missing physical tracking.
- **modality_dropout**: per example draw u; ifu<0.2zero pooled player channels0:24, if0.2<=u<0.4mask the whole PPbranch, otherwise keep both. Never drop both player branches in the same example. Ball channels24:29remain unchanged. Primary hypothesis: reduce provider-specific reliance on either player representation.
- **mixstyle**: training-only probability0.5, mix detached per-channel temporal mean/std of pooled player channels0:23across a random batch permutation. Beta(0.5,0.5) coefficient per example, variance population estimate plus1e-6. Visibility/ball channels23:onwards unchanged. No target/provider statistics used.
- **rdrop**: two stochastic forwards on the same batch. Mean of the two original short-state losses plus0.5times symmetric joint H/A/D KL over known frames. The joint distribution is factored from the two logits; no third head or extra optimizer step. At inference use ordinary deterministic eval network.

No regularizer, decoder or ensemble-weight scan after new scores become available.

## Fixed target observations and decoder

For every target match use two conditions:

- `clean`: original prepared observation condition. For IDSSE this name means the existing r0corruption, not originally clean tracking; for SkillCorner it means existing cleanpreparation.
- `gap`: additionally mask35%of previously visible ball rows in contiguous10..60sample spans, separately within half groups. Seed20261119plus deterministic match hash. Labels/timeline/players do not change. Recompute X16and every PP/ball observation feature from the masked ball; never retain stale original ball-distance features.

The same27matches produce54paired conditions, not54independent matches. Gap is a robustness diagnostic, not a fresh holdout. Source training uses original condition only; no target-gap data is used for training or normalization.

Decoder fixed to preceding dead45: threshold0.45,min dead1s, half/live resets, spatial evidence and one-second merge, full original18lambda grid calibrated after merging to **source-training-only** possession rate. No target-rate adaptation, public GTstatistics, prior shifts, learned ensemble or fixed kickoffframe rule.

Five recipes x27testmatches x2conditions =270decoded arrays and official metric rows. Main metric official mean spellF1 across IoU0.3/0.4/0.5/0.6/0.7, all H/Aspells including short; macroF1secondary. Cross-provider differences also include provider GTdefinitions/coverage and tracking quality, not just overfitting.

## Prediction seal before scoring

Prepare observation carriers, features and external label vault without metric evaluation. Semantic tests plus15real smoke seedfits (five recipes, three seeds,1update) on IDSSE source, target inference on one SkillCorner match in both conditions, then isolated saved-model replay. Smoke does not read target labels or calculate target scores.

After smoke, freeze source/PLAN, allsourceinput digests, observation/feature digests, external label-vault digests and guard policy. No old score reports are used to select a recipe. Guard-compatible source verification checks only inputs permitted for each role; target label hashes are checked by scorer after sealing.

Each full job fits source-only and stores checkpoint, metadata including actual names/statistics/parameters/budgets, then predicts every opposite-provider match in both conditions. Another process reloads each of10saved ensembles and reproduces every full probability and decoded array exactly. Replay proof stores all14or40condition IDs applicable to its target cohort.

Before any new target metric is computed, exclusively create one write-once seal committing all300artifacts:270predictionNPZs plus10checkpoints,10metadataJSONs and10replay-proofJSONs. Expected artifact IDs/paths are fixed and complete. Verify SHA256for every file and reject missing/duplicate/modified artifacts. Only after valid full seal may a fresh scorer process open external target label vault and calculate scores. This operationally blocks score-driven retraining within this campaign; it cannot undo adaptive history or determine a human's earlier intent.

Final separate-process verification rehashes the seal and independently recomputes all270metrics from the exact same sealed predictions and vault; results must agree. This is inference/decode/metric reproduction, not another full training refit.

## Primary decision, fixed before scoring

Primary **modality_dropout versus control**, on `clean` condition, equal mean of the two target-provider means. Require gain>=0.01, neither provider's mean delta below-0.02, and provider-stratified paired match-bootstrap95%lower>0. Bootstrap5.000draws, seed20261120. Draw matches independently within each provider and weight providers equally. Gap metrics and worst-case deltas reported but do not select another primary.

If primary fails, keep control for this research comparison. Report all other recipes as exploratory, no automatic raw-winner substitution. Nominal intervals are not corrected for multiple comparisons or historical adaptation. Do not turn a reused dataset into a blind claim by renaming the split.

## Delivery boundary

Code/docs and aggregate JSON under provider_generalization and sibling results prefix; raw carriers, vault, weights, predictions, spans and seals remain ignored cache/out. No competition public/private evaluation, target CSV, automatic commit or push. Report direct-public read prevention, external source-target separation and prior exposure separately. Absence of observed forbidden reads is not evidence that the resulting model is guaranteed free of overfitting.
