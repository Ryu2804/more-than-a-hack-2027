# Common Benchmark

Shared evaluation harness for all branches. See `../../docs/repo_inventory.md` and the plan in `~/.opencode/plan/repo-consolidation-common-benchmark.md`.

## Purpose
- Evaluate any branch on the same IDSSE matches, degradation, split, and official scorer.
- Prevent unfair comparison and public-window selection.

## Expected inputs
- Ground truth CSV: `frame,label` (IDSSE full-match possession labels).
- Prediction CSV: `frame,label`.
- Match id and split assignment.

## Outputs
- Per-match metrics JSON.
- Aggregated metrics JSON/CSV.
- Candidate registry update.