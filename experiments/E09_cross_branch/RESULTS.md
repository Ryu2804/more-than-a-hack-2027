# Cross-branch: the primary passes; a simpler ensemble scores the highest mean

All experiments completed: three new contextual models, 7 LOSO folds, 63 fresh neural seed fits / 31,500 updates, 28 probability/decoder settings, 196 per-match evaluations and 7 native-rule diagnostics. No new CPU SetSeq fits, old weights, public GT or target submissions. [PLAN](PLAN.md), [adaptation of each branch's source](SOURCES.md), [how to run](README.md).

## Decision frozen before the run

The primary `triple_combined/hysteresis` passes the promotion rule **for research**:

| Pipeline | Mean IDSSE spellF1 | Mean macroF1 |
|---|---:|---:|
| Control short-loss PP-TCN/dead45 | 0.595755 | 0.902935 |
| Combined primary of all components | **0.609030** | **0.909290** |
| Best raw alternative: short + SetSeq25/dead45 | **0.615458** | **0.909587** |

The primary uses 50% short-loss PP-TCN, 25% SetSeq and 25% combined contextual TCN with Jeremy's contact features + Rasya's touch-memory/dead cues. The contextual TCN backbone uses Ryu's PP encoder, 3 seeds and the same 500-update budget as the previous benchmark. Hysteresis low .35 / high .50 as in the PLAN.

Primary gain over the control **0.013274**, paired-bootstrap 95% **[0.005245, 0.021057]**, wins **6/7 matches**. The three conditions set before the run are met: gain ≥ .005, ≥ 5 wins, lower bound > 0. This is not evidence of a private improvement; the matches and hypotheses had been exposed historically.

The simpler best raw alternative uses **75% short-loss PP-TCN + 25% SetSeq** with the dead45 decoder. Mean 0.615458, gain 0.019702, interval [0.015308, 0.024763], wins 7/7. This setting was in the grid before the results were known, but it does not replace the primary post hoc. It is kept as an exploratory alternative worth further validation. There is no claim that the best of 28 settings is an unbiased held-out score.

**Combining every feature is not automatically best.** The new components help some combinations, but two complementary cached models give the highest result. This fact is not hidden to justify extra training.

## Adaptations actually made

- Ryu: the actual PP-TCN implementation, player-set attention, view symmetry and conditional dead/team probabilities. The shipped 27-match weights are not used because they include every IDSSE benchmark match.
- Jeremy: true 25-fps contact globals from the adopted touch-only lineage exp3/4/5, plus the SetSeq OOF probabilities from the previous benchmark. This is not the exact exp5 model or the exp7 recovery. The contact globals are learned inside the GPU TCN rather than by repeating the CPU GRU.
- Rasya: last-touch carried team, recency exp(−age/10 s), sustained out/slow cue with short-dead suppression of 6 s. All computed on raw 25-fps observations with team metadata, without GT orientation/dead/X mask. Speed from position / elapsed seconds, not divided by 3.6.
- Farras: the short-spell-loss OOF model, fixed joint ensembles, pre-registered dead45 versus hysteresis. The provider GT stays original, including all short spells.
- Main/consolidation: the official scorer and the shared-evaluation contract. Old adapter rankings, and dead-share numbers mislabelled as public spellF1, are not used.

Raw 25 fps is regenerated from cached clean observations with the canonical r0 corruption seed. Labels are removed before regeneration; sampled positions/ball/team/timeline are checked to equal the prepared 5-fps data. All seven matches pass. Contact/memory add sub-step information between the 5-fps samples: a feature-information augmentation, not a claim of identical input tensors with the old baselines.

## Full ablations

All numbers below are mean official spellF1 over the 7 held-out matches. No GT short spells are removed and the label definition is unchanged.

| Probability family | Dead45 | Hysteresis |
|---|---:|---:|
| short control | 0.595755 | 0.595994 |
| SetSeq | 0.578602 | 0.580111 |
| PP-TCN | 0.571090 | 0.570414 |
| contact-only TCN | 0.579219 | 0.579810 |
| memory-only TCN | 0.581164 | 0.583277 |
| contact+memory TCN | 0.586845 | 0.584343 |
| short75+SetSeq25 | 0.615458 | 0.614183 |
| short75+PP25 | 0.593436 | 0.594299 |
| short75+contact25 | 0.596079 | 0.597846 |
| short75+memory25 | 0.596968 | 0.597868 |
| short75+combined25 | 0.597035 | 0.597924 |
| short50+SetSeq25+contact25 | 0.605333 | 0.604767 |
| short50+SetSeq25+combined25 | 0.610665 | 0.609030 |
| visibility-weighted combined+short | 0.598298 | 0.599283 |

The three fresh contextual arms share the 36-channel architecture, PP encoder, seeds, 500 updates and sampling. Contact-only zeroes the 3 memory channels, memory-only zeroes the 4 contact channels, combined uses all 7 channels. Compared with the cached 29-channel baseline there are differences in information/dimension/loss: do not attribute the whole difference to a single feature.

Negative findings are kept:

1. None of the three standalone contextual TCNs beats the short-loss control on the mean.
2. Adding PP25 to short does not help; an ensemble is not a guarantee.
3. The quality/visibility blend does not reach the primary's minimum gain.
4. Hysteresis is almost flat on the short control and lowers several combinations. Wave6's high public diagnostic does not guarantee that decoder is better on IDSSE.
5. The primary with dead45 reaches 0.610665, slightly above the hysteresis primary 0.609030; the primary's decoder was not changed after seeing this.

## Native Rasya-style diagnostic

The rule computed from corrected raw 25 fps and sampled to 5 fps reaches mean spellF1 **0.322034**. This is not Rasya's original 0.5263/0.5057 and not the shared probabilistic decoder. The raw source/temporal resolution, the 6-s dead suppression, the label-free orientation and the missingness differ from the previous fair adapter. No ablation isolates each rule change. The neural models receive the cue/memory as features they can ignore, not as a hard override from a still-weak rule.

## Data split and selection

Every learned fresh model in each fold trains on exactly the other 6 matches. The original short, SetSeq and PP probabilities are used only for the held-out identity with a matching original manifest; no final model is inferred on training games. Normalisation and decoder rate come from the training fold only. No learned meta-model, label-based provider gate or weights optimised on outer labels.

The primary was fixed before the full fits; all 28 settings are still reported. Bootstrap 5,000 draws, seed 20261115, unit = unique match, not frame/seed. The interval is unadjusted, and this is adaptive research on the same seven matches; not a nested tuning evaluation or fresh external confirmation. No claim of beating every team member's exact submission.

## Verification and runtime

- 31 semantic tests pass: sub-step ball impulse, earliest tie, missingness/half isolation, team symmetry, touch duration/recency, correct metric speed, dead suppression at 149/150 frames, strict regeneration identity, joint fusion, post-merge count calibration and the full 18-lambda grid.
- The first prepare failed because of a duplicate `seed` metadata key. The audit composition was fixed before the full-fit registry; prepare for all 7 matches and the actual GPU smoke test of all three models then finished with exit 0. No source/PLAN/input edits after the registry.
- The 9 smoke seed fits use 1 step, are kept separate from the 63 full seed fits and are not used as performance estimates.
- Full fits ran in stages, each family/fold in its own child process. All 21 training jobs exited 0. The actual metadata shows 6 unique training names, test excluded, 3 seeds / 500 steps; not just the intended split written down.
- Aggregate time inside the training jobs 1,424.704 s, excluding some startup/orchestration overhead. SetSeq was not retrained on CPU; only the verified OOF cache is used.
- The original control mean **0.5957554001098855** is reproduced exactly.
- An independent-process rebuild verifies **203 identical decoded arrays**: 196 model-setting/match outputs and 7 native diagnostics. This is a decoding rebuild, not a refit.

Manifest SHA: `dce422c8eb38801630c4909f648274e0b793ceadf5de001e614ab57b9c71b572`.

Evidence: [all 196 scores](results/cross_branch_results.json), [protocol](results/cross_branch_protocol.json), [source regeneration](results/cross_branch_preparation.json), [actual training membership](results/cross_branch_training.json), [checks](results/cross_branch_checks.json), [independent rebuild](results/cross_branch_verification.json).

## Output status

The primary passes for research; the best simpler ensemble remains an exploratory alternative. No new public evaluation, new target submission CSV or private-score claim. All new code is separate in cross_branch (now `E09_cross_branch/`); old campaigns and teammates' source are intact. Raw tracking, per-frame labels/probabilities, controls, native arrays and weights stay ignored. Only source/docs and 6 aggregate JSONs are exported.
