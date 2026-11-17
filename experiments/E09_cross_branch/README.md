# E09: Cross-branch cues (contact, possession memory, temporal ensembles)

[PLAN.md](PLAN.md) fixed the primary `triple_combined/hysteresis` before the full fits, compared with the original fair control `short/dead45`. Teammates' experiment results/caches and old campaigns are not changed.

Result: primary **IDSSE spellF1 0.609030**, control 0.595755, gain 0.013274, wins 6/7; it passes the research promotion rule. The best raw alternative, short75+SetSeq25/dead45, is higher at 0.615458 and wins 7/7, but stays exploratory. 31 semantic tests, 63 fresh seed fits and 203 identical decode rebuilds. No new public evaluation. Record card: [CARD.md](CARD.md). Full report: [RESULTS.md](RESULTS.md).

## Adaptations

- Ryu: PP-TCN, view symmetry, player-set attention, conditional dead/team probabilities.
- Jeremy: contact globals from 25-fps observations (the adopted touch-only recipe of exp3/4/5) and the cached SetSeq OOF from the shared benchmark.
- Rasya: label-free last-touch hold, contact recency, sustained out/slow cue, suppression of dead runs < 6 s. No GT flipping, oracle dead/X mask, or wrong speed unit.
- Farras: cached short-loss PP-TCN, fixed hysteresis decoder and joint-probability ensembles.
- Main/consolidation: the official scorer and common evaluation; the old non-comparable adapter rankings are not copied.

This transfers research components; it is not an exact reproduction of each branch's best submission. The cached SetSeq is not the original touch-exp5 model. The contact/memory features were moved into the GPU TCN to avoid repeating the long CPU SetSeq training; their effect is measured by three ablation models with the same architecture, input-channel count and budget.

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/cross_branch_results.json`:

   ```bash
   python experiments/check_results.py E09
   ```

   Expected: `PASS` for primary 0.609030, delta vs control +0.013274, promoted for research, best raw short75+SetSeq25/dead45 0.615458.

2. **Run the semantic tests** (contact/memory features, regeneration identity, fusion, count calibration):

   ```bash
   python -m pip install lxml        # needed by the vendored Jeremy parser
   python -B -m pytest -q experiments/E09_cross_branch
   ```

   Expected: `31 passed` (plus subtests).

## Reproduce the campaign (needs the E08 caches and clean 25-fps records)

The seven IDSSE r0 matches are the same as in fair_idsse (E08), with 7 LOSO folds. Each new model trains on the other 6 matches, 3 seeds 0/1/2, 500 updates/seed, batch 16, crop 384. Contact-only, memory-only and combined use 36 input channels and the same PP encoder; ablated channels are zero-filled. 63 fresh seed fits / 31,500 updates. Cached baseline probabilities are taken only on the held-out match, according to the six-match-exclusion metadata, manifest and input hashes.

Raw 25-fps observations are re-corrupted with the canonical r0 seed; labels are dropped before regeneration. Sampled positions/ball/metadata/timeline must match the prepared 5-fps data. The new controls add information between the sampled frames: a feature augmentation, not tensor-input parity with the old baselines.

Fourteen probability families × 2 fixed decoders = 28 settings / 196 per-match evaluations. The native raw-25-fps rule, sampled to 5 fps, is reported separately. Only the primary can pass research promotion (gain ≥ .005, ≥ 5/7 wins, paired-bootstrap 95% lower > 0); the best raw alternative does not automatically replace the primary. No public evaluation or new target submission. Data and hypotheses were already exposed; not a fresh blind test.

From the repository root, Python 3.10 with the dependencies/environment of fair_idsse. Required: local clean 25-fps records, canonical r0 packs, PP feature packs, and the complete, immutable fair_idsse OOF results/caches. No data is downloaded and no old weights are opened.

```bash
python -B -m unittest discover -s experiments/E09_cross_branch -p 'test_*.py' -v
python -B experiments/E09_cross_branch/prepare.py
python -B experiments/E09_cross_branch/check.py
python -B experiments/E09_cross_branch/run.py register
python -B experiments/E09_cross_branch/run.py fit-fold --match J03WMX
python -B experiments/E09_cross_branch/run.py all
python -B experiments/E09_cross_branch/run.py rebuild
```

`check.py` actually trains the three new models for 1 step/seed on the 6 real training matches and predicts the other match. This is a smoke test and is not mixed with the 500-step performance estimates. The parent treats a non-zero process exit as a failure even if the output was written. Fits use a separate child process per family/fold. `all` verifies/reuses only jobs whose exact identity matches and refuses to overwrite complete results. Source/PLAN/prepared controls are frozen before the full fits; never update a hash to hide a change.

`rebuild` reruns the 203 decoded arrays in another process (196 model-setting rows + 7 native-rule rows) from the frozen probabilities/context. It is a decoding rebuild, not a model refit.

What to compare after a rerun: the control mean must reproduce exactly 0.5957554001098855, `rebuild` must report 203 identical arrays, and the ablation table must match [RESULTS.md](RESULTS.md).

## Artifacts and publication limits

`cache/` holds per-frame controls/native predictions; `out/` holds audits, the pre-registration, models/probabilities, full results and verification. All ignored. Only code, documents and the aggregate JSON in `results/cross_branch_*.json` are shared. Raw tracking, labels/probabilities, weights and submissions are gitignored.

IDSSE CC BY 4.0; earlier provenance and credit: [fair benchmark](../E08_fair_idsse/README.md), [Jeremy reference](../E04_wave3/reference/SOURCE.md), [Ryu reference](../E05_wave4/reference/SOURCE.md). Full source refs are frozen in the registry; notes on source decisions are in the PLAN.

Branch/commit details, adopted versus exploratory results, and the parts deliberately not imported: [SOURCES.md](SOURCES.md).
