# E03: Wave2, gradient boosting + Viterbi on Metrica

The second Farras campaign: 16 method families (see [`methods/`](methods/)), 48 configurations and 192 external evaluations, under the same protocol as wave1 (Metrica Games 1/2, two leave-one-game-out folds, original + occluded; Game 3 after the freeze). It tests whether learned tree models with a Viterbi switch cost beat the wave1 rule.

Result: selected `w08_symmetric_boost/cost6` (histogram gradient boosting with H/A swap augmentation, Viterbi cost 6). External mean spellF1 0.517438, Game 3 0.599664 (original) / 0.433982 (occluded), public diagnostic 0.460606. The highest public diagnostic (0.560000, `w14_feature_ablation/control`) was **not** adopted, because it was not the external choice. Superseded by E04. Record card: [CARD.md](CARD.md). Full results and the wave1/2 history: [RESULTS.md](RESULTS.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers**:

   ```bash
   python experiments/check_results.py E03
   ```

   Expected: `PASS` for `w08_symmetric_boost/cost6` and external mean spellF1 0.517438 in `results/wave2_metrics.json`.

2. **Run the decoder tests**:

   ```bash
   python -m pip install -r experiments/E03_wave2/requirements.txt
   python -B -m pytest -q experiments/E03_wave2
   ```

   Expected: `7 passed`.

## Reproduce the campaign (needs the Ryu tracking cache and Metrica sample data)

Wave2 needs wave1's selection as a frozen comparison, so run [E02](../E02_wave1/README.md) first. From the repository root, Python 3.10:

```bash
python -m pip install -r experiments/E03_wave2/requirements.txt
python -B experiments/E03_wave2/prepare_holdout.py
python -B experiments/E03_wave2/test_decoders.py
python -B experiments/E03_wave2/run.py all --workers 4
python -B experiments/E03_wave2/run.py holdout
python -B experiments/E03_wave2/run.py public
```

`all` refuses to overwrite a frozen selection. To run a single family without changing the campaign choice, for example the configuration with the 0.56 public diagnostic:

```bash
python -B experiments/E03_wave2/run.py method --path experiments/E03_wave2/methods/w14_feature_ablation.py
```

What to compare:
- The selection must be `w08_symmetric_boost/cost6`.
- `results/holdout_metrics.json` must be reproduced.
- The target CSV must hash to `45e03662fccbeb238e9df723c99829df60e3d8d641d539be9f6d1e1fd59b0446`.
- `results/relocation_verification.json` records that the rerun after moving the code reproduced every target CSV hash and every external/public metric.
