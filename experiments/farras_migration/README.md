# Farras migration archive: wave2 adapter

The final team method is **Farras wave5 `short_gru25/dead45`**, not this migration.
Use [FINAL_METHOD.md](../../docs/FINAL_METHOD.md), [final_method.json](../../docs/final_method.json),
and [BRANCHES.md](../../docs/BRANCHES.md) for the report. Wave5 source and CSV are in the
[pinned Farras release](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60);
this directory does not supply its runtime or exporter.

`predict.py` preserves a best-effort IDSSE adaptation of wave2
`w08_symmetric_boost/cost6` (HGB + Viterbi), with environment-specific source paths.
Its [common-benchmark results](../CANDIDATES.md#historical-common-benchmark-idsse-7-matches)
are historical adapter results, not a measurement of the exact final method.

## Original migration plan

The earlier README described a pending scaffold and proposed:

- Replace Metrica training with IDSSE degraded inputs.
- Port HGB + Viterbi to per-match prediction.
- Run the common benchmark runner.

That pending-scaffold note belongs to the original plan; an adapter and benchmark summary
are now retained in this archive. Their presence does not establish raw-data reproduction
or submission readiness for wave5.
