# Provider generalisation results

## Conclusion on public-overfitting risk

**There is still no basis to guarantee the model is free of overfitting.** The inspected old code chose candidates from external-development metrics and then opened the public GT as a diagnostic after the freeze. That does not prove that already-seen public scores never influenced later research. All 27 external matches were also exposed in earlier campaigns. Audit and file/line evidence: [EXPOSURE.md](EXPOSURE.md).

The new campaign tightens the process: no access to competition public/private labels; fresh source-provider-only training; match/provider disjoint per direction; label-free target inference; every prediction committed before any new target score is computed. This reduces leakage paths and score-driven iteration **within this campaign**; it does not erase the adaptive history or produce a fresh blind test.

The new regularisers do not improve generalisation enough to meet the rule. The control is kept. A drop compared with the old IDSSE ensemble numbers does not directly prove public overfitting: the new protocol uses another provider as the training source, a standalone model, a source-rate-only decoder and different labels/coverage. The 0.624577 of the older campaign is not an equivalent comparison.

## Experiment and primary decision

Train 7 IDSSE r0 → test 20 SkillCorner clean; train 20 SkillCorner clean → test 7 IDSSE r0. Fresh PP-TCN, 135,219 parameters/network, three seeds, 500 updates, batch 16, crop 384. All five recipes use the original short loss and 50% event sampling; normalisation, event indices and rate calibration come only from source training. Total **30 seed fits / 15,000 optimiser updates**. RDrop uses two stochastic forwards per update, so compute is not identical.

The primary `modality_dropout` gives an equal-provider original spellF1 of **0.5303783116699301**, versus control **0.5408640368900999**. Difference **−0.010485725220169892**, provider-stratified paired-bootstrap 95% CI **[−0.016208310154948818, −0.004824001196390026]**. Delta on target IDSSE −0.02193845247146054, on target SkillCorner +0.0009670020311207562. It fails the gain ≥ 0.01 requirement, fails the −0.02 minimum provider guard, and the interval does not support an improvement. Not promoted.

The best raw exploratory `rdrop` reaches **0.5408823298349871**, only **+0.00001829294488716563** over the control; CI [−0.0016194618724505068, 0.001493495988374656]. Practically the same; not evidence of improvement. The primary is not replaced by the raw winner.

Bootstrap 5,000 draws, seed 20261120, paired match resampling separately within each target provider, then equal provider weights. Seven IDSSE and 20 SkillCorner matches are not 27 equal-weight rows for the main objective. All intervals are nominal, not corrected for multiple comparisons/adaptive history. The gap diagnostics do not pick a replacement model.

## All transfer results

`original` means the carrier already available: IDSSE stays r0 corruption, SkillCorner stays the clean preparation. `gap` adds a 35% mask to previously visible ball rows, with spans of 10–60 samples and a fixed seed; X16 and all PP/ball features are recomputed. The 54 conditions come from the same 27 matches, not 54 independent matches. Main metric: official mean spellF1 across IoU 0.3/0.4/0.5/0.6/0.7, over all H/A spells including short ones.

| Method | SC→IDSSE original | IDSSE→SC original | SC→IDSSE gap | IDSSE→SC gap | Equal mean original | Equal mean gap |
|---|---:|---:|---:|---:|---:|---:|
| RDrop | 0.503927 | 0.577837 | 0.480469 | 0.560447 | 0.540882 | 0.520458 |
| Control | 0.505107 | 0.576621 | 0.479137 | 0.559069 | 0.540864 | 0.519103 |
| PP token dropout | 0.503630 | 0.572568 | 0.480268 | 0.560256 | 0.538099 | 0.520262 |
| Modality dropout | 0.483168 | 0.577588 | 0.462700 | 0.555926 | 0.530378 | 0.509313 |
| MixStyle | 0.493595 | 0.558371 | 0.469814 | 0.551363 | 0.525983 | 0.510589 |

| Method | Original delta [95% CI] vs control | Gap delta [95% CI] vs control | Original wins/27 |
|---|---|---|---:|
| RDrop | +0.000018 [-0.001619, 0.001493] | +0.001355 [-0.002065, 0.004565] | 15 |
| Control | 0 [0, 0] | 0 [0, 0] | 0 |
| PP token dropout | -0.002765 [-0.007056, 0.000416] | +0.001159 [-0.001007, 0.003391] | 9 |
| Modality dropout | -0.010486 [-0.016208, -0.004824] | -0.009790 [-0.014294, -0.005668] | 9 |
| MixStyle | -0.014881 [-0.018800, -0.010874] | -0.008514 [-0.013077, -0.004344] | 1 |

Modality dropout improves slightly on target SkillCorner but drops on target IDSSE. MixStyle drops on both providers. PP dropout/RDrop show slightly better gap point estimates, but each interval still crosses 0. There is no score-driven reason to adopt any of them from these results.

[INFERENCE] Fully dropping one player representation can remove information that the target provider still needs; mixing statistics can also disturb informative feature geometry. This experiment does not isolate those mechanisms. The negative results are recorded, not hidden as implementation failures.

## Source-only decoder statistics

| Training source | Training matches | Source possession spells/minute | Source dead share |
|---|---:|---:|---:|
| IDSSE | 7 | 4.581112 | 0.448819 |
| SkillCorner | 20 | 2.591973 | 0.466170 |

No target possession-rate or dead-share calibration. [INFERENCE] The difference in spell density / proxy labels between sources can affect the decoder under transfer. Tracking quality, provider label definitions and domain shift are mixed in this evaluation; a score drop cannot be attributed to overfitting alone.

## Evidence of read isolation and of predictions before scoring

1. Preparation copies the external labels into a vault and builds 54 observation/feature carriers without y/y25. Preparation and registration compute input digests, not metrics. No provided-game carrier or competition GT is opened.
2. Fifteen real smoke seed fits use the 7 source IDSSE matches, 1 update/seed. Smoke inference and saved-model replay test one SkillCorner match in two conditions, without target labels or metrics.
3. Ten full source-only ensembles are trained with separate source/test providers. The fit refuses the target provider's original labelled packs/features and the whole vault. Predict/replay refuse all original labelled packs and the vault; only observation-only carriers are used.
4. All 270 full target probability arrays **and** decoded arrays are reproduced identically from the ten saved models in another process. Nothing is retrained to fit a score.
5. The write-once seal binds **300 artifacts**: 270 prediction files, 10 checkpoints, 10 metadata files, 10 replay proofs. Full coverage and SHA-256 must match before the scorer may open the external target-label vault. Test scoring happens only after the seal is complete.
6. The runtime guard refuses a deliberate public-GT probe before any byte is read. Probes are separate from unintended forbidden reads. A trusted Python open policy is not an OS sandbox; native I/O, existing descriptors or adversarial bypasses are not fully covered. `successful_forbidden_reads=0` in the hook report is a property of that mechanism, not an independent security audit.

This proves the process that ran and the integrity of the checked artifacts; it does not prove the model is free of overfitting or will improve on the private target match.

## Verification, pre-freeze fixes and timing

59 semantic tests completed: **58 passed, 1 skipped**. Coverage includes the math/gradients/masking of the regularisers, actual subprocess read denial, seal tampering/missing artifacts, rejection of label-bearing carriers, seal-before-vault ordering and the provider-balanced bootstrap. The detailed reason for the skip is not stored in the aggregate runner; no test was removed and no numeric tolerance was loosened.

A separate verification process completed: **270 identical metric rows**, 300 sealed artifacts matching, 270 identical probability/decoded replays from 10 saved-model ensembles. Of 50 runtime reports: **140 intentional denied probes, 0 unexpected forbidden-read attempts**. This is evidence of the read policy and the reproduction that ran, not a guarantee of being overfit-free.

Before the manifest/full fits, the initial checks found a relative-path fixture crossing Windows drives and floating-point cancellation in MixStyle float32. The fixture was moved to a temporary directory on the same drive as the cwd; the MixStyle mixing arithmetic was made float64 while the output keeps the input dtype. Source enumeration was also fixed so it does not try to hash PyTorch's synthetic `_classes.py` as a real file. All fixes happened before the source freeze and before full training; no configuration was changed after scores were seen.

Sum of the 10 fit-job timers **819.110 s**, including source loading, fitting and checkpoint/metadata preparation after input verification. It excludes preparation, the final check (351.5 s), startup/source verification, initial target inference/replay, seal/scoring/recompute and orchestration overhead. The independent metric verification finished in 159.30 s. CUDA GPU jobs ran sequentially. Equal FLOPs are not claimed, because RDrop uses two forwards and MixStyle uses float64 mixing arithmetic.

Source/PLAN/input manifest SHA-256: `6ac1740c290d3ab332b33eb7a0a1ab6452c23bd6b285835cd1c3ba1c84039279`.
Prediction-seal SHA-256: `6429e7462a2faf3353a914c29e99ed160aaf953d99e08fb267d40021474272c2`.

Aggregate JSON in `results/provider_generalization_*.json`. Observation carriers, the label vault, tracking, weights, predictions, mask spans and the raw seal stay in the ignored cache/out. No new public/private score, target CSV. Commands and pipeline limits: [README.md](README.md).
