# Experiments

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
- Final submission: copy the chosen `out/*.csv` to `<team>.csv`; the notebook must regenerate it.
- `_template/NOTES.md`: hypothesis / assets used or dropped / result / verdict for the technical document.
