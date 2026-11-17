# E13: Provider generalisation (source-provider-only transfer)

A source-provider-only generalisation test, with target labels kept out of the fit/inference processes and every prediction sealed before scoring. This **does not guarantee freedom from overfitting**: all 27 external matches have been exposed before. Historical audit: [EXPOSURE.md](EXPOSURE.md). Recipes, split and decision are frozen in [PLAN.md](PLAN.md).

Result: control equal-provider spellF1 0.540864; the primary `modality_dropout` scores 0.530378 (−0.010486) and is **rejected**; RDrop 0.540882 is practically equal to the control. Record card: [CARD.md](CARD.md). Full results and decision: [RESULTS.md](RESULTS.md).

## What is different

Train on 7 IDSSE r0 → test on 20 SkillCorner clean, then train on 20 SkillCorner clean → test on 7 IDSSE r0. Fresh weights for each direction; no SetSeq/OOF mixtures, shipped weights or target-rate calibration. Normalisation, event indices, short weights and decoder statistics are source-only. Source and target matches are disjoint in each direction, but they are not historically fresh blind matches.

Five recipes on the same 135,219-parameter PP-TCN: the event-sampler control, time-tied PP token dropout, one-player-branch modality dropout, training-only MixStyle on pooled features, and two-forward RDrop joint consistency. Each 3 seeds / 500 updates / batch 16 / crop 384. Total 30 seed fits / 15,000 updates. RDrop uses two forward graphs per update, so compute is not identical.

Each target match is scored in its original condition (`clean`, which for IDSSE is actually the existing r0) and with an additional 35% ball drop (`gap`). X16 and the PP/ball features are regenerated from the masked observations. The 54 conditions come from 27 matches, not 54 independent matches. The dead45 decoder is fixed, with the rate from source training only.

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/provider_generalization_results.json`:

   ```bash
   python experiments/check_results.py E13
   ```

   Expected: `PASS` for control 0.540864, modality_dropout 0.530378, delta −0.010486, RDrop 0.540882, not promoted, 27 target matches.

2. **Run the semantic tests** (read guard, seal, dropout/MixStyle/RDrop, consistency, pipeline):

   ```bash
   python -B -m pytest -q experiments/E13_provider_generalization
   ```

   Expected: `58 passed, 1 skipped` (plus subtests).

## Reproduce the campaign (needs the 27 external packs and a GPU)

From the repository root, Python 3.10 with the fair_idsse/event_learning environment. The 27 local external packs with their original PP features are needed. No raw data is downloaded, the provided target match is not used, and the competition GT is not opened.

```bash
python -B experiments/E13_provider_generalization/prepare.py
python -B experiments/E13_provider_generalization/check.py
python -B experiments/E13_provider_generalization/run.py register
python -B experiments/E13_provider_generalization/run.py all-job --source idsse --family control
python -B experiments/E13_provider_generalization/run.py all
python -B experiments/E13_provider_generalization/run.py verify
```

`prepare.py` separates the observation carriers from the external-label vault without computing any metric. The new cache uses its own folder; source packs and old campaigns are not changed. Reads of the competition public GT are refused by the runtime guard, including a deliberate probe that reads no bytes.

`check.py` runs the semantic tests, then 15 real smoke seed fits of 1 update each on the 7 source IDSSE matches. Inference and an isolated saved-model replay use one SkillCorner match in two conditions through the label-free carriers. The target-label vault and the original label-bearing test packs are refused. The smoke test computes no metric.

`register` freezes the source/PLAN/input digests after the checks. `all-job` runs fit, predict and replay in three different processes. The fit can only open source-labelled packs. Predict/replay use observation-only carriers and cannot open the original labelled packs or the vault through the guarded Python `open`. GPU jobs run sequentially. `all` completes all 10 jobs, seals everything and then scores. Do not score partially or rebuild a recipe after seeing the results.

For staged orchestration after the 10 jobs finish:

```bash
python -B experiments/E13_provider_generalization/run.py seal
python -B experiments/E13_provider_generalization/run.py score
python -B experiments/E13_provider_generalization/run.py verify
```

The write-once seal covers 270 prediction NPZs, 10 checkpoints, 10 metadata files and 10 replay proofs. All 300 artifacts must be complete with matching hashes. The scorer verifies the seal before opening the external-label vault. `verify`, in another process, recomputes the 270 metric rows from the sealed predictions; it is not a full training refit. The full model replay already requires 270 identical joint-probability/decoded arrays. Completed results are not overwritten.

What to compare after a rerun: the seal SHA-256 `6429e7462a2faf3353a914c29e99ed160aaf953d99e08fb267d40021474272c2` and manifest SHA-256 `6ac1740c290d3ab332b33eb7a0a1ab6452c23bd6b285835cd1c3ba1c84039279` (if inputs are identical), 270 reproduced metric rows, and the recipe table in [RESULTS.md](RESULTS.md).

## Decision and limits of the evidence

Primary `modality_dropout` versus `control`, original condition, objective = equal mean over the two target providers. Promotion requires a gain ≥ 0.01, no provider dropping more than 0.02, and a provider-stratified paired-bootstrap 95% lower bound > 0. Bootstrap 5,000 draws, seed 20261120. If it fails, the control is kept; the gap diagnostic and the raw winner do not pick a replacement primary. Every recipe is fixed, with no public-target tuning.

The runtime guard is a trusted Python read policy, **not an OS sandbox**. It does not close all native I/O, existing descriptors or adversarial bypasses. Expected probes are reported separately from unexpected blocked reads; the latter are fatal. The hash seal guarantees the integrity of the verified artifacts; it does not prove the absence of overfitting. Provider transfer also changes label/proxy definitions, tracking quality and coverage. The nominal bootstrap does not correct for adaptive history or multiple comparisons.

Aggregates in `results/provider_generalization_*.json`. The ignored cache/out store the label vault, observations, features, weights, predictions, stress spans and the seal; that raw/derived data is not published. No target CSV, new public/private evaluation.
