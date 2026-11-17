# E07: Wave6, conservative fusion and occlusion-aware decoding

[Audit of every branch](BRANCH_REVIEW.md) and the [plan before evaluation](PLAN.md). Six paths, 36 alternatives plus the exact wave5 control. No new training seed fits: the wave5 OOF/final probabilities are used frozen, to isolate fusion/decoder changes. The official scorer and all short GT stay the same.

Result: the best hysteresis variant reaches development spellF1 **0.674807**, public diagnostic **0.786207**, public macroF1 0.925369. The development gain of 0.004707 fails the 0.005 promotion threshold, so the selection stays wave5. The best alternative wins 6/6 paired stress cases, but IDSSE drops slightly and long-spell recall has not recovered. 40 regression cases pass after one test-import fix; the selected decoding rebuild gives 199,319 identical rows. Record card: [CARD.md](CARD.md). Full report: [RESULTS.md](RESULTS.md). No private improvement is claimed.

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/wave6_*.json`:

   ```bash
   python experiments/check_results.py E07
   ```

   Expected: `PASS` for:
   - best alternative `m05_dead_hysteresis/low35_high50`
   - development 0.674807, gain 0.004707
   - gain below 0.005 and not promoted
   - bootstrap lower bound 0.003412
   - public 0.786207

   Together these show why the higher public score was not adopted.

2. **Run the unit tests** (blends, visibility weights, heads, occlusion cost, dead hysteresis, short guard):

   ```bash
   python -m pip install -r experiments/E07_wave6/requirements.txt
   python -B -m pytest -q experiments/E07_wave6
   ```

   Expected: `40 passed` (plus subtests).

## Reproduce the campaign (needs the frozen wave3–5 caches)

From the repository root, Python 3.10 and the dependencies of the earlier campaigns. Required: cache/data/features, the frozen wave3..5 manifests and selections, including cached short-TCN, boundary-TCN, old PP and GRU predictions on all development/holdout/stress/target cases. There is no fallback that regenerates missing predictions or writes to old campaigns. The target stays 199,319 frames.

```bash
python -m pip install -r experiments/E07_wave6/requirements.txt
python -B -m unittest discover -s experiments/E07_wave6 -p 'test_*.py' -v
python -B experiments/E07_wave6/run.py dev
python -B experiments/E07_wave6/run.py holdout
python -B experiments/E07_wave6/run.py rebuild
python -B experiments/E07_wave6/run.py public
```

`dev` refuses to overwrite a frozen selection. Before evaluation, source/PLAN/input hashes are registered. Later stages verify hashes rather than accepting a cache just because the file exists. Evaluation runs in one CPU process with threadpool 1; no CUDA training is repeated. Torch is still needed by the inherited modules. The old training backend in the manifest is provenance, not a claim of wave6 training.

`rebuild` must run as a separate process after holdout. It reloads the probabilities, repeats the selected decoding, writes a temporary CSV to compare its SHA-256 and then deletes it. It is an independent decoding rebuild, **not a learned-model refit**. Model-refit evidence is inherited from wave5/wave3.

What to compare after a rerun:
- `out/selection.json` keeps the control (wave5) selected.
- The development table matches [RESULTS.md](RESULTS.md).
- `rebuild` reports the selected CSV hash `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.

## Protocol

- Development 23 scenarios / 21 unique matches. Training-fold probabilities/rate stay from the 19 IDSSE+SkillCorner matches, 6 grouped folds. Metrica 1/2 zero-shot; not retrained.
- Objective: mean spellF1 per match, per provider, then the equal mean of three providers. MacroF1 tie-breaker.
- Promotion: at least +0.005, at least 2/3 providers winning, no provider dropping > 0.01, lower paired-bootstrap 95% > 0. Bootstrap 5,000, seed 20261114, unique matches within provider. The interval is exploratory and not corrected for the multiple search or the adaptation of older campaigns.
- Old holdout: 9 unique matches / 10 ordinary scenarios, plus 6 paired SkillCorner stress cases from the same matches. All previously exposed. Not fresh blind validation.
- Only selected, best_new and previous_wave5 were tested after the freeze; identical roles do not count as independent models. The public window's 17 GT possessions are a format/alignment audit only, not a basis for selection. The private score is unknown.

## Artifacts

`out/` (ignored): preregistered.json, selection.json, execution.json, holdout_report.json, duration_diagnostics.json, verification.json, public_audit.json and candidates/*.csv. Source predictions are not copied into the new campaign. `results/wave6_*.json` holds the committed aggregates.

Only code, documents and aggregate metrics are safe to share. Raw tracking, per-frame labels/probabilities, weights and submission CSVs are gitignored. Credit and source licenses are inherited from [E05_wave4/reference/SOURCE.md](../E05_wave4/reference/SOURCE.md): SkillCorner MIT and IDSSE CC BY 4.0, Metrica only as an evaluation proxy for the selected model.
