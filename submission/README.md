# Submission package: wave5 `short_gru25/dead45`

The team's final method is `short_gru25/dead45` from [E06](../experiments/E06_wave5/). Its specification is [docs/FINAL_METHOD.md](../docs/FINAL_METHOD.md) (facts as data in [final_method.json](../docs/final_method.json)). The runtime identity, CSV hashes and selection evidence are locked in [method.json](method.json).

The deliverables are at the repository root: [`carried_by_admin.csv`](../carried_by_admin.csv) and [`carried_by_admin.pdf`](../carried_by_admin.pdf) (see the root [README](../README.md)).

## The two CSVs

| File | SHA-256 | Public spellF1 / macroF1 | Role |
|---|---|---:|---|
| `artifacts/e06_retrained.csv` = root `carried_by_admin.csv` | `850e6d25d96ea051d0f746548ab59de9b4c04fdededca53fff16fbe9c2053b5c` | 0.772414 / 0.901002 | **Submitted.** The method retrained from scratch by [`E06_wave5.ipynb`](../experiments/E06_wave5/E06_wave5.ipynb) ([receipt](../experiments/E06_wave5/results/wave5_retrain_verification.json)) |
| `artifacts/farras_wave5.csv` | `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8` | 0.772414 / 0.902059 | Frozen campaign candidate. The selection evidence describes this file; reference only |

The two files agree on 99.52% of graded frames. Both have 199,319 rows (`frame,label`, 25-fps tactical timeline). Neither is ground truth.

## Exporting the submitted CSV

From the repository root:

```bash
python -m pip install -r requirements.txt
python -B submission/export.py                                  # writes submission/out/farras.csv
python -B submission/export.py --output carried_by_admin.csv    # the root deliverable
```

The command checks the hashes of the selection receipt, the retrain receipt and the CSV, runs the official validator, and copies `artifacts/e06_retrained.csv` byte for byte. It prints `SHA256 850e6d25…`. It does not train, score, read GT or re-select a candidate. A different existing target file is never overwritten; repeating an export to an identical file is safe.

To rebuild the CSV from the raw match bundle instead, run the E06 notebook ([E06 README](../experiments/E06_wave5/README.md), level 3).

## Method and reasons for the choice

- Probability mixture: 75% short-loss PP-TCN and 25% set-sequence BiGRU (wave3 recipe). Complete live spells shorter than 4 seconds get double state-loss weight; crop/unknown fragments are not treated as complete short spells.
- Decoder `dead45`: dead threshold 0.45, minimum dead 1 second, H/A DP reset in every live interval, possession count calibrated after merging.
- Gate 0.35 m, kappa 0.05, tau 3, merge 1 second, rate scale 1. No dead-prior shift and no target-specific kickoff rule.
- Wave5 met the external adoption rule fixed before training. The public score was not the basis for selection. Wave6 has a higher public score, but its development gain did not reach the promotion threshold; it remains a research comparison.

| Recorded evaluation | Value | Evidence |
|---|---:|---|
| Development spellF1, equal provider mean (frozen candidate) | 0.670099 | [Wave5 verification](../experiments/E06_wave5/results/wave5_verification.json) |
| Development gain over wave4 | +0.013609 | Wave5 verification |
| Public diagnostic spellF1, submitted file | 0.772414 | [Retrain receipt](../experiments/E06_wave5/results/wave5_retrain_verification.json) |
| Public diagnostic macroF1, submitted file | 0.901002 | Retrain receipt |

The private/full-match score is unknown. Development, the public diagnostic and the IDSSE research benchmarks use different protocols; their numbers are not one leaderboard.

## Limitations

- A rerun of the notebook on other hardware or library versions is expected to agree closely with the submitted file, not byte for byte.
- External development and holdout scores were measured on the frozen candidate, not on the retrained file.
- The paired ball-drop stress is slightly worse than wave4 and the possession under-count is not solved.
- The external matches have been exposed historically; there is no evidence of being overfit-free or of superiority on the private GT.

History and negative results are in the [experiment registry](../experiments/REGISTRY.md).

## Verification records

[verification.json](verification.json) and [publication_verification.json](publication_verification.json) are historical records of the export checks on the **frozen candidate** (`cd59fb90…`), made before the retrained file existed: validator accepts 199,319 rows, idempotent repeated export, refusal to overwrite a different file, refusal of a changed candidate, and export from a minimal snapshot without data, caches, weights or GT. They are kept unchanged as evidence. The submitted file is checked by `python experiments/check_results.py E06`.
