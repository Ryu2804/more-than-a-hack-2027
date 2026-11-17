# E11: Spell training (spell-balanced loss, event sampling, dual-scale TCN)

New training experiments for short possessions: a spell-balanced team loss, short-event-centred sampling, and a shared-kernel dual-scale TCN. This folder is separate from other members' experiments and from every earlier campaign. [PLAN.md](PLAN.md) fixes the configurations, the primary and the promotion rule before the full fits.

Result: the primary `event_replace25/dead45` reaches 0.618941 against the control's 0.615458 (+0.003484, wins 5/7). The gain is below the 0.005 threshold, so it is **not promoted**. The event-sampled standalone TCN improves on the original short-loss TCN (0.608023 vs 0.595755). Record card: [CARD.md](CARD.md). Full results: [RESULTS.md](RESULTS.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/spell_training_results.json`:

   ```bash
   python experiments/check_results.py E11
   ```

   Expected: `PASS` for primary 0.618941, delta +0.003484, not promoted, best exploratory `event_replace25/consensus` 0.619563.

2. **Run the semantic tests** (duration/censoring, weighted denominators, event enrichment, RNG parity, dual-scale gradients):

   ```bash
   python -B -m pytest -q experiments/E11_spell_training
   ```

   Expected: `27 passed` (plus subtests).

## Reproduce the campaign (needs the E08–E10 caches and a GPU)

From the repository root, Python 3.10 with the fair_idsse environment, PyTorch/Numba/threadpoolctl, and the same canonical r0 packs and PP feature packs. Also needed: the cached short_tcn/SetSeq OOF with fair_idsse provenance, the cross_branch controls and the temporal_methods results/rebuild. No data is downloaded and no predecessor model weights are opened.

```bash
python -B experiments/E11_spell_training/check.py
python -B experiments/E11_spell_training/run.py register
python -B experiments/E11_spell_training/run.py fit-fold --match J03WMX
python -B experiments/E11_spell_training/run.py all
python -B experiments/E11_spell_training/run.py rebuild
```

`check.py` runs the semantic tests, then 9 actual seed fits of 1 update each on the six real training matches, plus full held-out inference for the three families. Three separate processes reload the smoke checkpoints and check that the probabilities are identical. This is a smoke test, not a performance estimate.

`register` freezes the source/PLAN/input/provenance hashes after the checks. `fit-fold` runs each model family in a separate child process, three seeds / 500 updates per family; another process then loads the new weights and repeats the full held-out inference. A non-zero exit is still a failure even if artifacts were written. GPU jobs run sequentially; parallel development does not mean concurrent GPU training. `all` resumes only jobs whose exact identity/hash/budget match and does not overwrite completed results.

`evaluate` can be called after all seven folds finish; it scores the 20 setting combinations on every match. `rebuild` rebuilds the 140 decoded arrays in another process and verifies the 21 saved-model inference proofs. It is not a second full training refit.

What to compare after a rerun: `rebuild` reports 140 identical arrays and 21 identical inference replays; the control is byte-identical; the manifest SHA-256 is `2db5b8e793cfd43559425d600aa282422558e135016178944b3c8bfaf3d306ca`; the configuration table matches [RESULTS.md](RESULTS.md).

## Comparison

Control: short-loss TCN 75% + SetSeq 25%, dead45, mean official IDSSE spellF1 0.6154575861548992. New primary: event_replace25/dead45, i.e. short 50% + SetSeq 25% + event-sampled TCN 25%. Each family is also scored pure, base75new25 and replace25, with the frozen dead45 and the frozen consensus decoder. All combinations were fixed before the run, not scanned after seeing results.

Same 7 LOSO folds / 6 training matches, 3 seeds, 500 updates, batch 16, crop 384, 29 channels and parameter count. Dual-scale uses the same kernel for dilation 1 and the original dilation; no extra parameters, more convolution compute. The spell loss and event sampling do change the objective/window distribution; there is no claim of full tensor/cost parity.

Primary promotion: gain ≥ 0.005, wins ≥ 5/7 matches, paired match-bootstrap 95% lower > 0. If it fails, the control is kept; the best raw exploratory configuration is still reported. The matches and hypotheses had been exposed; not a fresh blind test. No new public/private evaluation or target submission.

## Artifacts

Aggregate JSON in `results/spell_training_*.json`. `out/jobs/` stores the new weights/probabilities/metadata, `out/replays/` stores the exact inference proofs. Raw training data, labels, weights and probabilities are not published; all of `out/` is ignored.

Component provenance: [fair_idsse](../E08_fair_idsse/README.md), [cross_branch](../E09_cross_branch/SOURCES.md), [temporal_methods](../E10_temporal_methods/README.md). The base PP-TCN follows Ryu's reference, kept separate from its original source; the actual feature builder keeps the correct per-player opponent mask. Every original source is immutable.
