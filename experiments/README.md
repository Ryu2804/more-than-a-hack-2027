# Experiments

## Final team method

The team selected **Farras wave5 `short_gru25/dead45`**. Report authors should use the [canonical specification and limitations](../docs/FINAL_METHOD.md), [report data](../docs/final_method.json), and [branch/archive map](../docs/BRANCHES.md).

This directory preserves Rasya's research experiments and their original candidate decisions. It does not contain the selected wave5 runtime. The exact selected code and artifact remain at the [pinned Farras release](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60); local commands below describe historical workflows, not final-team reproduction instructions.

## Research archive workflow

Each experiment is a notebook in the Three Makasih layout (header + Ringkasan, Metodologi, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    00_eda.ipynb                 data understanding (no tuning on the public GT)
    001_baseline/001_baseline.ipynb
    002_possession_dp/run.py     possession DP with gated dead-ball evidence (public window: sanity only)
    003_possession_layers/run.py soft ownership, ball-in-flight, Team tag prior, identity repair, HSMM
    004_video_team_check/run.py  jersey-colour check of the ATD team labels from the video (label-free)
    005_possession_colour_owner/run.py  colour-derived ownership as the DP emission
    <NNN_name>/<NNN_name>.ipynb  one folder per idea; writes only to its own out/ (gitignored)

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Historical candidate export: copy the chosen `out/*.csv` to `<team>.csv`; the notebook must regenerate it. This is the original local workflow, not the selected team package.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.
