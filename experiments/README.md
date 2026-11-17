# Experiments

The final team method is **Farras wave5 `short_gru25/dead45`**.
Report authors must use the [final specification](../docs/FINAL_METHOD.md),
[Typst facts](../docs/final_method.json), and [branch/archive map](../docs/BRANCHES.md).
The [submission package](../submission/README.md) contains the frozen CSV export and readiness status.
All other primary/control decisions below are campaign-local research, not a replacement of the final method.

Notebook experiments use the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)
    exp_farras/                scripted campaigns, frozen evidence, and research history

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- The primary wave5 CSV has a pinned hash and a dedicated export command; do not copy a newer or higher-public-scoring research artifact in its place. Raw-bundle reproduction is not yet complete.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.
