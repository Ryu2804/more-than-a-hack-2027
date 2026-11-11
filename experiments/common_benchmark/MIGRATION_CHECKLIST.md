# Migration Checklist to Common Benchmark

## General (all branches)
- [ ] Obtain IDSSE data and set `MTAH_DATA_DIR` or place in `data/`.
- [ ] Generate degraded inputs using the frozen degrader (jeremy `judge/` snapshot).
- [ ] Produce per-match prediction CSVs (`<MATCH>_pred.csv`).
- [ ] Run `run_benchmark.py` and store outputs in `results/`.
- [ ] Update `../CANDIDATES.md`.
- [ ] Add smoke test in experiment `tests/`.

## Branch-specific
### jeremy
- [ ] Export SetSeq predictions per IDSSE match (already in exp_1 protocol; adapt paths).
- [ ] Re-run with common benchmark runner instead of private judge code.
- [ ] Record proxy + confirm metrics.

### rasya
- [ ] Port touch-hold + gated dead cue to IDSSE per-match inference.
- [ ] Produce predictions for DEV + CONFIRM.
- [ ] Run benchmark and compare to jeremy.

### farras
- [ ] Replace Metrica training with IDSSE (or keep as exploratory only).
- [ ] Port HGB + Viterbi to per-match prediction.
- [ ] Run benchmark.

### ryu
- [ ] Port TCN to IDSSE or run zero-shot with calibration.
- [ ] Produce predictions per match.
- [ ] Run benchmark and document dead recall.