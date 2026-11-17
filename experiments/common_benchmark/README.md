# Common Benchmark

Historical consolidation benchmark, retained with its original results and adapters. The final
team method is **Farras wave5 `short_gru25/dead45`**; see
[FINAL_METHOD.md](../../docs/FINAL_METHOD.md), [final_method.json](../../docs/final_method.json),
and the [branch/archive map](../../docs/BRANCHES.md). This harness is not the wave5 runtime.

## Limits of the historical comparison

- Farras used the adapted wave2 HGB + Viterbi method, not the selected wave5 short-TCN/GRU mixture.
- Ryu's shipped weights had already seen IDSSE benchmark matches during training; this is train/test exposure.
- Rasya's external adapter used GT-assisted orientation mapping and an unknown-label (`X`) mask. Its results are not an inference-only comparison.
- Jeremy used an older exp_1 adapter, not every later campaign candidate.

Consequently, the seven-match aggregate cannot rank the exact final candidates or establish a
blind, leakage-free comparison. Preserve its numbers as historical adapter results, not as
evidence that any method beats the selected wave5 pipeline.

## Original purpose

The consolidation plan aimed to evaluate branches on common IDSSE matches, degradation,
splits, and scorer, avoiding public-window selection. That goal does not remove the exposure
and implementation differences above. The original plan path was
`~/.opencode/plan/repo-consolidation-common-benchmark.md`; it is not a portable report source.
The [old repository inventory](../../docs/repo_inventory.md) is also a historical snapshot.

## Expected inputs
- Ground truth CSV: `frame,label` (IDSSE full-match possession labels).
- Prediction CSV: `frame,label`.
- Match id and split assignment.

## Outputs
- Per-match metrics JSON.
- Aggregated metrics JSON/CSV.
- Candidate registry update.