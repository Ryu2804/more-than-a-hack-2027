# Experiments

The primary submission candidate on branch `farras` is **Farras wave5 `short_gru25/dead45`**.
Use [submission/README.md](../submission/README.md) for the frozen CSV export and readiness status.
Later research campaigns do not automatically replace it.

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
