# Experiments

## Final team method; local research archive

The final team method is **Farras wave5 `short_gru25/dead45`**, not a local Jeremy variant.
Use [FINAL_METHOD.md](../docs/FINAL_METHOD.md), [final_method.json](../docs/final_method.json),
and [BRANCHES.md](../docs/BRANCHES.md) for the report. Wave5 runtime and CSV are in the
[pinned Farras release](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60);
this branch's code does not become that pipeline through documentation alignment.

`exp_1` through `exp_7` retain their original code, reports, and evidence. Any primary,
control, adopted, rejected, or final designation there belongs to that experiment's campaign,
not the final team selection. In particular, exp7 was not adopted as the team method.
Read those reports for research history and attribution, not as the wave5 implementation spec.

Public diagnostics, external development metrics, and prediction dead shares are different
quantities. Final selection does not imply raw-bundle reproduction or a blind/private result;
the canonical documents record the limits.

## Historical experiment workflow

Each experiment is a notebook in the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Historical candidate export: experiments wrote their own `out/*.csv`; these files are not the final team artifact. Use the pinned wave5 CSV and readiness notes in the canonical documents.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.
