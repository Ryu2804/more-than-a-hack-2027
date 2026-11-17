# E10: Temporal methods (duration HSMM, consensus change-point, boundary relocation)

Three new decoding methods on top of the short-loss TCN 75% + SetSeq 25% ensemble: an explicit-duration HSMM, a consensus change-point, and ordered boundary relocation. Old campaigns and other members' experiment folders are not changed. [PLAN.md](PLAN.md) freezes the hypothesis, configurations and decision rule before the new evaluation.

Result: the primary HSMM (`duration/strength0.5_bias0`) scores 0.605838 against the control's 0.615458 (−0.009620, wins 1/7) and is **rejected**. The best exploratory consensus reaches 0.616970 (+0.001512), with an interval crossing 0. Record card: [CARD.md](CARD.md). Full results: [RESULTS.md](RESULTS.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/temporal_methods_results.json`:

   ```bash
   python experiments/check_results.py E10
   ```

   Expected: `PASS` for primary 0.605838, delta −0.009620, not promoted, best consensus 0.616970.

2. **Run the semantic tests** (exhaustive optimality of the HSMM and boundary DPs, censoring, half resets, contact agreement, decoder parity):

   ```bash
   python -B -m pytest -q experiments/E10_temporal_methods
   ```

   Expected: `47 passed` (plus subtests).

## Reproduce the campaign (needs the E08/E09 caches)

From the repository root, Python 3.10 with the fair_idsse/cross_branch environment. Required: the original r0 packs, the cached short_tcn/setseq OOF from fair_idsse, the 25-fps controls and the complete cross_branch provenance. Old weights are not opened and no network is retrained.

```bash
python -B experiments/E10_temporal_methods/check.py
python -B experiments/E10_temporal_methods/run.py all
python -B experiments/E10_temporal_methods/run.py rebuild
```

`check.py` runs the semantic tests, including exhaustive tiny-path proofs for the two dynamic programmes, and then smoke-tests all 19 settings on 96 real observations from two halves. The smoke test computes no metric. `all` verifies the predecessor, freezes the source/PLAN/input hashes, fits duration priors from the six training matches of each fold and evaluates 133 setting-match pairs. Finished results are not overwritten; a new campaign must use a new folder. `rebuild`, in a separate process, refits the seven duration priors and rebuilds the 133 decoded arrays from the frozen inputs.

What to compare after a rerun: `rebuild` reports 133 identical arrays and 7 identical priors; the table in [RESULTS.md](RESULTS.md) is reproduced; the manifest SHA-256 is `38829cd0f0f48a1ffffa2d4dd0cca57c14dcc7f684a95b58eed0e697b0076b75`.

## Reading the results

The baseline is the best previous exploratory result, mean official IDSSE spellF1 0.6154575861548992. New primary: HSMM strength 0.5 / bias 0.0. Promotion rule: gain ≥ 0.005, wins ≥ 5/7 matches and paired match-bootstrap 95% lower > 0. All 18 new variants are reported, but a raw winner does not automatically replace a failed primary. Every match has been exposed; the bootstrap does not correct for the adaptive history or multiple comparisons. Not a fresh blind test.

The primary metric keeps all H/A spells and the mean over IoU 0.3/0.4/0.5/0.6/0.7 via the official scorer. The IoU-0.75 recall per duration is diagnostic only; it is not the main metric. Every source uses the same folds, 25-fps expansion and unknown masking. The duration priors learn from raw labels of the six training matches, not from OOF predictions of other folds that might involve the held-out match. No held-out label calibration or oracle orientation/D/X mask.

Aggregate JSON is shared in `results/temporal_methods_*.json`. Raw priors, decoded hashes and working reports are stored in the ignored `out/`. No target submission, new public evaluation.

## References

[Yu 2010: Hidden semi-Markov models](https://www.sciencedirect.com/science/article/pii/S0004370209001416) for explicit-duration modelling; [Ishikawa 2021: boundary-aware temporal segmentation](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html) as the motivation for boundary-focused decoding. The implementation here is our own adaptation, not a reproduction of either paper's architecture or results. Model/data provenance: [fair_idsse](../E08_fair_idsse/README.md) and [cross_branch](../E09_cross_branch/SOURCES.md).
