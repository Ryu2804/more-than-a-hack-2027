# Experiments

## Final team method

The team selected **Farras wave5 `short_gru25/dead45`**.
Use the [final specification](../docs/FINAL_METHOD.md), [Typst facts](../docs/final_method.json),
and [branch/archive map](../docs/BRANCHES.md). Experiments and baselines here are research history;
their primary/control or final labels do not replace the team selection.
The pinned Farras release contains the candidate runtime and CSV; this branch does not.

Each experiment is a notebook in the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Final team artifact and reproduction limitations: [FINAL_METHOD.md](../docs/FINAL_METHOD.md). Do not substitute an arbitrary experiment's `out/*.csv`.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.
