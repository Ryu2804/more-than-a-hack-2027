# Experiments

The final method is wave5 `short_gru25/dead45` ([E06](E06_wave5/)), described in
[docs/FINAL_METHOD.md](../docs/FINAL_METHOD.md) (facts as data in [final_method.json](../docs/final_method.json);
branches in [BRANCHES.md](../docs/BRANCHES.md)).
The [submission package](../submission/README.md) holds the submitted CSV and its export tool.
All other primary/control decisions below are campaign-local research, not a replacement of the final method.

**All experiments in run order, with one record card each: [REGISTRY.md](REGISTRY.md).**
Each experiment is one folder `EXX_<name>/` with its code, `results/`, a `CARD.md` record card and a `README.md` with step-by-step instructions to test its result.
Check every experiment's recorded numbers without match data: `python experiments/check_results.py` (or `... E06` for one).
New experiments take the next ID and start from [_template/CARD.md](_template/CARD.md).

Notebook experiments use one layout (header with the team, then Summary, Methodology, Environment,
numbered `[n]` sections: markdown intro, code, markdown reading of the result).

    E00_eda/E00_eda.ipynb            data understanding (no tuning on the public GT)
    E01_baseline/E01_baseline.ipynb  official baseline
    E02_wave1/ ... E13_*/            scripted campaigns: code, PLAN/RESULTS, results/*.json, CARD.md
    EXX_<name>/                      one folder per idea; writes only to its own out/ and cache/ (gitignored)
    CAMPAIGNS.md                     inputs, environment and reproduction notes shared by E02-E13

- `common.py`: `cached()` (parsed-tracking cache in `_cache/`), `save()`, `score()`.
- Run: `jupyter nbconvert --to notebook --execute --inplace <notebook>` from any dir inside the repo.
- Public 5-minute score is a pipeline sanity check only. Never pick between variants with it (RULES.md).
- Final team artifact and reproduction limitations: [FINAL_METHOD.md](../docs/FINAL_METHOD.md). Do not substitute an arbitrary experiment's `out/*.csv`.
- `_template/NOTES.md`: quick run notes inside an experiment folder; the record card ([_template/CARD.md](_template/CARD.md)) is the scientific record.
