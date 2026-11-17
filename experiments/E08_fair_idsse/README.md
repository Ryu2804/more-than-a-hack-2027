# E08: Fair IDSSE benchmark of model families on a shared split

[PLAN.md](PLAN.md) fixes the data, budget and scoring before the full fits. Five families: HGB symmetric, SetSeq, PP-TCN, short-loss PP-TCN, and label-free touch-hold. This is an adapted-family benchmark, not an exact reproduction of team members' submissions.

Result: mean spellF1 short-loss PP-TCN **0.595755**, SetSeq 0.578602, PP-TCN 0.571090, HGB 0.315719. The short loss wins on the mean, but the interval of its comparison with SetSeq still crosses 0. 32 semantic tests pass; 63 new neural seed fits and 7 HGB fits completed, and 63 decoded arrays were reproduced identically in a separate process. Record card: [CARD.md](CARD.md). Full results and limits: [RESULTS.md](RESULTS.md).

Every arm uses the same seven prepared IDSSE r0 matches byte for byte, with 7 leave-one-match-out folds. Each learned arm is retrained from scratch on the other 6 matches. No pretrained checkpoints, public GT, target predictions or hyperparameter selection from the outer fold. Original provider labels, including short possessions, are used by the official scorer. The matches had been exposed in earlier research; this is not fresh blind validation.

The four learned arms use the same decoder (wave5 dead45), plus the same argmax diagnostic. Touch-hold is in a native-rule panel because it outputs hard labels; do not claim the rule receives an identical probabilistic decoder. Neural seeds 0/1/2, 500 updates/seed, batch 16, window 384; an equal-update budget, not equal FLOPs or the 1,500-step production recipe. Optimiser/augmentation/representation differ and are recorded in the PLAN.

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/fair_idsse_results.json`:

   ```bash
   python experiments/check_results.py E08
   ```

   Expected: `PASS` for short-loss PP-TCN 0.595755, SetSeq 0.578602, PP-TCN 0.571090, HGB 0.315719.

2. **Run the semantic tests** (bridge, classic HGB/rule, neural arms):

   ```bash
   python -B -m pytest -q experiments/E08_fair_idsse
   ```

   Expected: `32 passed` (plus subtests).

## Reproduce the benchmark (needs the prepared IDSSE packs)

From the repository root, Python 3.10 with the wave5 dependencies; the prepared wave3 IDSSE packs and wave4 feature packs for all seven matches are needed. Nothing is re-downloaded and no old cache is changed.

```bash
python -B -m unittest discover -s experiments/E08_fair_idsse -p 'test_*.py' -v
python -B experiments/E08_fair_idsse/check.py
python -B experiments/E08_fair_idsse/run.py all
python -B experiments/E08_fair_idsse/run.py rebuild
```

`check.py` really fits/predicts on real data with 1 update/seed for the three neural arms, plus HGB/rule. The smoke result is not a performance estimate. `all` runs 63 neural seed fits / 31,500 updates and 7 HGB fits; all 63 score rows must complete. New caches carry the fold identity, training names, config and manifest SHA; a mismatch is rejected. `all` refuses to overwrite complete results but can resume identical finished jobs if an earlier process was interrupted. Do not change source/PLAN/hashes to accept a mismatched cache.

Each family/fold runs in a separate child process. SetSeq uses CPU because its CUDA child still exited fatally after computing; both PP-TCNs use CUDA, HGB/rule CPU. The parent only scores and accepts a cache after the child exits 0. The backend per family is bound in the registry. The update/seed/data budget is unchanged; do not use these results for latency or equal-FLOP efficiency rankings.

`rebuild` runs in a separate process, loads the probability outputs and repeats every decoder, comparing 63 label-array hashes. It is a decoding rebuild, not a refit. It produces no public submission and does not promote any candidate.

What to compare after a rerun: `out/results.json` must give the same per-family means as `results/fair_idsse_results.json`, and `rebuild` must report 63 identical decoded hashes.

## Output and sources

`out/` (ignored) stores preregistered.json, the combined/isolated smoke tests, job probabilities/metadata/checkpoints, results.json and verification.json. The combined smoke test that exited fatally is kept as evidence of the failure, not as a pass. Only per-match metrics, means, paired bootstrap and source/input digests are exported to `results/fair_idsse_*.json`. Do not share tracking/label/probability arrays or weights.

IDSSE CC BY 4.0; team Home/Away metadata and corruption provenance in wave3/cache/idsse_audit.json. Code/architecture/data credit: [wave3 source](../E04_wave3/reference/SOURCE.md), [wave4 source](../E05_wave4/reference/SOURCE.md). Both PP arms use the actual builder `E05_wave4/features.py`, which computes opponents relative to the player's team. The earlier audit claim that the bug in Ryu's reference file was automatically inherited by our model is corrected in the PLAN; the reference file and the model builder are not the same function.
