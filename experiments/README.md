# Experiments

## Final team method

The team selected **Farras wave5 `short_gru25/dead45`**. Report authors should use the [canonical specification and limitations](../docs/FINAL_METHOD.md), [report data](../docs/final_method.json), and [branch/archive map](../docs/BRANCHES.md).

This directory preserves Ryu's research experiments and their original candidate decisions. It does not contain the selected wave5 runtime. The exact selected code and artifact remain at the [pinned Farras release](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60); local commands below describe historical workflows, not final-team reproduction instructions.

## Research archive workflow

Each experiment is a notebook in the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Historical candidate export: copy the chosen `out/*.csv` to `<team>.csv`; the notebook must regenerate it. This is the original local workflow, not the selected team package.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.

## Historical Ryu pipeline

`python experiments/015_final_pipeline/run.py` regenerates the historical Ryu candidate (see [pipeline notes](015_final_pipeline/NOTES.md), [historical technical document](../TECHNICAL_DOCUMENT.md)). Notebooks 013 (supervised model) and 014 (decoder) are the research behind it; [RESEARCH.md](RESEARCH.md) collects the literature findings. The directory name `015_final_pipeline` reflects that campaign's selection, not the final team method.
