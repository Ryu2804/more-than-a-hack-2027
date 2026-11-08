# Experiments

Each experiment is a notebook in the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Final submission: copy the chosen `out/*.csv` to `<team>.csv`; the notebook must regenerate it.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.

## Final pipeline

`python experiments/015_final_pipeline/run.py` regenerates the submission (see `015_final_pipeline/NOTES.md`, `../TECHNICAL_DOCUMENT.md`). Notebooks 013 (supervised model) and 014 (decoder) are the research behind it; `RESEARCH.md` collects the literature findings.
