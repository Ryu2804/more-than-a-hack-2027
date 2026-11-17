# Audit of every branch, fetched 2026-11-14

`git fetch --all --prune` succeeded. Six remote branches were found; origin/HEAD is not an extra branch. The local main and farras branches matched their remote refs at the time of the inventory. No checkout, merge, commit or push. The audit reads source/reports and aggregate artifacts; it does not rerun the team's full pipelines.

| Remote branch | Commit | What was checked |
|---|---|---|
| main | ef22d025b0c02bcdbcec53df34973ece0f53a701 | Baselines and the scorer specification |
| farras | 87f3236d4ed875f4cc868d9f750a0776a7779333 | Pushed wave1/2; compared with local wave3/4/5 |
| jeremy | ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1 | REPORT/decision/source of exp_1 to exp_7 |
| rasya | a13249724b994728b5686d01bc48e6647cd1e40e | CONTEXT, TECHNICAL, parser, cues and pipeline 012 |
| ryu | 5eaf3e595e91fa0502aac48037d0cae8c356ec6f | NOTES 015/016/017, features, training, shipped decoder |
| consolidation | 264efa67e1a2e74cce22fea81ed784acf159b258 | Registry, shared scorer/runner, the four members' adapters and the summary |

## Most important findings

### 1. The consolidation registry mislabels some numbers

`experiments/CANDIDATES.md` lists 0.408/0.410/0.488 as Jeremy's exp2/3/5 public spellF1. Jeremy's branch source shows those numbers match the **target dead share**, not a score against the public GT:

- exp2 `results/target_candidate.json`: D share about 0.407543, `public_labels_used:false`.
- exp3 `results/target_candidate.json`: dead_share 0.41040684389104665, `public_labels_used:false`.
- exp5 `results/target_candidates.json`: dead_share 0.48837004349824054, `public_labels_used:false`.

The public spellF1 actually recorded in Jeremy's report is exp1 **0.581818**, macroF1 **0.725033**. The registry needs to distinguish prediction statistics, proxy evaluation and public evaluation. The wrong registry numbers are not copied into the wave6 comparison.

### 2. The common benchmark is not an equal ranking of the latest pipelines

The consolidation summaries record mean IDSSE scores of Jeremy 0.598011, Ryu 0.572393, Rasya 0.520453, Farras 0.333041. These four numbers do not prove a quality order of the latest models:

- `farras_migration/predict.py` is still wave2, with training changed to the first two IDSSE matches that are not the target (`[:2]`), not the original Metrica protocol or the cross-provider wave5. The README still says the scaffold is pending although a summary exists. The remote Farras folder did not yet contain the local wave3..5.
- `ryu_migration/predict.py` uses the shipped `weights_nometrica` for IDSSE matches that were already in its 27-match training. It is not a held-out checkpoint for the seven benchmark matches. It has a 0.407 prior shift and a 25-fps interpolation that differs from Farras's adapter.
- `jeremy_migration/predict.py` trains `setseq_noaug`, while the registry names boundary-aware SetSeq as the best. It is an adapter, not a reproduction of the latest exp7 candidate.
- `rasya_migration/predict.py` uses a touch context that depends on held-out GT for team orientation and the X mask.
- `common_benchmark/config.json` still has the degrader hash as `TBD`; the runner does not read that config, accepts an arbitrary list of matches and defaults to bootstrap 0. `score_wrapper.py` takes labels by order without validating that the frame IDs of the two CSVs match. It needs an input/split/fit manifest and alignment validation before it can serve as a shared benchmark.

No common benchmark was rerun in wave6. The consolidation branch source was not changed silently. The wave6 comparison uses the same local OOF/protocol as wave5 and the official scorer, with a control that must reproduce the old score exactly.

### 3. Ryu's nearest-opponent feature is wrong for opposite-view players

`origin/ryu:experiments/features.py:174-190`: the candidate mask `~same` is based on the view team, not on player j's team. A player from the opposite-view team is in its own candidate set, so its nearest distance is 0. The local snapshot `wave4/reference/features.py` is identical in this part; our learned features inherit that behaviour as well.

The fix must make the opponent mask relative to `T[j]`, exclude the player itself, then rebuild the features and retrain every related fold/final checkpoint. Changing only inference would create a training/inference feature mismatch. Wave6 deliberately does **not** claim this bug is fixed: all old probabilities/checkpoints stay frozen, and the fusion/decoder effects are tested separately.

Ryu's code also calibrates differently in validation and in the shipped pipeline: `017/remote_pipeline.py` stores a match-weighted training rate; the shipped `015/run.py` uses a source-balanced rate and a 0.407 prior shift. Revalidation must test the same deployment pipeline, not only the decoder primitive. `--retrain` in 015 only trains on 3 Metrica matches; it does not reproduce the shipped 27-match model. TECHNICAL_DOCUMENT still describes the old model/numbers.

### 4. Rasya's branch has a unit bug and depends on held-out labels

`origin/rasya:experiments/012_spellf1/barath.py:79-82` divides the ATD speed by 3.6, whereas `DATA.md:85-86` says it is in m/s. The 2 m/s "slow" cutoff effectively becomes 7.2 m/s on the real input. Ryu's 017 NOTES report this cue at 0.014 as written and 0.400 after the fix; that is a team report, not repeated in wave6.

`012_spellf1/touch.py:31-47` uses agreement with the full held-out GT to flip teams; `deployable.py:49-56` copies the GT X mask. Team mapping and the graded area must come from metadata before the inference can be called label-free. CONTEXT records 0.5263 dev / 0.5057 confirm; longslow 0.5504/0.5413 was rejected by the internal adoption threshold. The confirm pair has been used repeatedly. TECHNICAL describes the old 002, not 012; the evidence vault and gate.py are not fully tracked.

Rasya's pipeline is not imported into wave6 production. The teammate's source stays intact.

### 5. Jeremy's exp7 reduces simulated damage but does not pass adoption

`origin/jeremy:experiments/exp_7/REPORT.md`, `results/decision.json` and `summary.json`: measured-view provider spellF1 0.677863, gain over the clean reference 0.013727 with interval [0.003328, 0.026526], 6/7 wins. But all three ctrl clauses fail: gain 0.002216 with an interval crossing 0; no-harm −0.005872 and stress −0.005444 exceed the −0.005 bound. There is no full-data exp7 model/candidate/rebuild; do not treat it as a ready submission.

The exp6 detector, precision@2m 0.760362 / recall 0.230371, fails the 0.90/0.50 gate. The earlier measured-recovery simulation lowered provider spellF1 by about 0.125. Exp7 uses test-derived detector noise/false-positive rates and target-derived coverage/run lengths; it is not an independent cross-source holdout. Touch-only exp4/5 help the proxy, but the exp5 target dead share becomes 0.488370 versus a prior of 0.407. Noisy ball recovery/touch is not added to wave6 without transfer validation.

Jeremy's exp7 hash verification checks prereg/judge/constants but does not enforce the full dictionary of source hashes; the evaluation cache is skipped based on file existence. This is a potential reproduction gap, not evidence that results were manipulated. Wave6 verifies source/input hashes and rejects missing probabilities rather than loading a cache just because it exists.

### 6. Our wave5 still has a false-transition trade-off

Wave5 mean development spellF1 is 0.670099 versus 0.656490 for wave4. Public spellF1 stays 0.772414, which is not a private score. Short recall rises, but pooled stress precision drops from 46.13% to 43.84%; recall of GT ≥ 15 s drops from 71.98% to 68.64%. Only 2/6 paired stress cases win. The ordinary holdout was already exposed and is not a new test.

Wave6 isolates conservative fusion/heads/occlusion costs/hysteresis/A-B-A guard with frozen probabilities. No new training or public tuning. The bootstrap uses unique matches as clusters and does not treat clean/stress conditions as separate matches. Selection and public phases are separate; the old candidate stays if the promotion rule fails.

## Limits of score comparisons

Ryu's 017 reports 0.679 on 30 grouped held-out matches and a full-model public score of 0.747; the default no-Metrica public score cannot be identified from NOTES alone. The local wave4 adapter once measured 0.760, but it is not a byte-exact reproduction of Ryu's source. Jeremy's 0.748 uses an L_ctrl that merges short spells, not the original provider GT. These numbers cannot be compared directly with the wave5/6 equal-provider objective.

The official main metric remains mean spellF1 over IoU 0.3/0.4/0.5/0.6/0.7 across all possessions. The official public baseline smooth50 is about 0.325581; the published full-match baseline is 0.2433. The public window has only 17 GT possessions. The private/full-match score of the team's model is unknown.

## Provenance

Source URLs: `https://github.com/Ryu2804/more-than-a-hack-2027/blob/<commit>/<path>` with the commit from the table. The source audit did not need to execute scripts from other branches. Files of wave1..5, main and the teammates' experiments were not changed. The only new updates are wave6 and the Farras index. This report does not fix the remote registry or push branches automatically.
