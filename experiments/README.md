# Experiments

## Final team method; consolidation archive

The final team method is **Farras wave5 `short_gru25/dead45`**. Report authors should use
[FINAL_METHOD.md](../docs/FINAL_METHOD.md), [final_method.json](../docs/final_method.json),
and [BRANCHES.md](../docs/BRANCHES.md). Source and CSV are in the
[pinned Farras release](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60),
not in this branch's old migration adapters.

Local experiments and migration code remain research/archive material, not automatically the
final pipeline. Primary/control and candidate decisions in their reports are campaign-local.
The [candidate registry](CANDIDATES.md) records the final choice separately from historical
results. The [common benchmark](common_benchmark/README.md) did not compare exact final
candidates: it used Farras wave2, exposed Ryu weights, Rasya GT-assisted external mapping,
and an older Jeremy adapter.

Do not treat public diagnostics as development scores, infer missing public scores from
prediction statistics, or describe this archive as raw-reproduction-ready or a blind evaluation.

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
