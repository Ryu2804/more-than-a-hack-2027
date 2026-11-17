# Team final method: Farras wave5

The team chose `short_gru25/dead45` as the final method. Report authors use the [cross-branch specification](../docs/FINAL_METHOD.md) and the [Typst facts](../docs/final_method.json). The candidate's runtime identity, CSV hash and selection evidence are locked in [method.json](method.json). A new campaign does not automatically replace this choice.

The submission deliverables themselves are at the repository root: [`carried_by_admin.csv`](../carried_by_admin.csv) and [`carried_by_admin.pdf`](../carried_by_admin.pdf) (see the root [README](../README.md)). The submitted CSV is this method **retrained from scratch** by the E06 notebook (`artifacts/e06_retrained.csv`, SHA-256 `850e6d25…`); the frozen campaign candidate stays in `artifacts/farras_wave5.csv` as the reference that the selection evidence describes. `export.py` checks the submitted file against both the selection receipt and the retrain receipt (`experiments/E06_wave5/results/wave5_retrain_verification.json`).

## Exporting the selected CSV

From the repository root, with Python 3.10 and the evaluation dependencies:

```bash
python -m pip install -r requirements.txt
python -B submission/export.py
```

Output: `submission/out/farras.csv`. For a different file name:

```bash
python -B submission/export.py --output submission/out/teamname.csv
python -B submission/export.py --output carried_by_admin.csv     # the root deliverable
```

The command checks the hashes of the selection evidence and the CSV, runs the official validator, and copies the frozen candidate byte for byte. It does not train, score, read GT or re-select a candidate. A different existing target file is never overwritten; repeating an export to an identical file is safe.

The selected prediction CSV is committed for harness checks:

```text
submission/artifacts/farras_wave5.csv
```

A fresh clone can read this CSV or run the export without caches, models, raw data or GT. The file is the output of the frozen wave5 algorithm, not ground truth. No other candidate CSV is published. The campaign procedure and the cache/model dependencies for re-training or re-inference are in [E06_wave5/README.md](../experiments/E06_wave5/README.md).

SHA-256 of the main CSV:

```text
cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8
```

## For the team harness

Use [artifacts/farras_wave5.csv](artifacts/farras_wave5.csv) as the BAR-ATH prediction: columns `frame,label`, 199,319 rows on the 25-fps tactical timeline. The machine-readable identity is in [method.json](method.json); configuration and results are in [`experiments/E06_wave5/`](../experiments/E06_wave5/).

Do not adapt this BAR-ATH file into predictions for external matches, and do not use the wave2 adapter as a stand-in for wave5. This package provides the exact target CSV for the harness and the source for inspection; it is not an inference API for new matches or evidence of blind validation.

## Method and reasons for the choice

- Probability mixture: 75% wave5 short-loss PP-TCN and 25% frozen wave3 hybrid GRU. Complete live spells shorter than 4 seconds get double state-loss weight; crop/unknown fragments are not treated as complete short spells.
- Decoder `dead45`: dead threshold 0.45, minimum dead 1 second, H/A DP reset in every live interval, possession count calibrated after merging.
- Gate 0.35 m, kappa 0.05, tau 3, merge 1 second, rate scale 1. No dead-prior shift and no target-specific kickoff rule.
- Wave5 met the external adoption rule fixed before training. The public score was not the basis for selection. Wave6 has a higher public score, but its development gain did not reach the promotion threshold; it remains a research comparison.

| Recorded evaluation | Value | Evidence |
|---|---:|---|
| Development spellF1, equal provider mean | 0.670099 | [Wave5 verification](../experiments/E06_wave5/results/wave5_verification.json) |
| Development gain over wave4 | +0.013609 | Wave5 verification |
| Public diagnostic spellF1 | 0.772414 | [Wave5 public audit](../experiments/E06_wave5/results/wave5_public.json) |
| Public diagnostic macroF1 | 0.902059 | Wave5 public audit |

These scores are campaign records, not a new public evaluation. The private/full-match score is unknown. Development, the public diagnostic and the IDSSE research benchmarks use different protocols; their numbers are not one leaderboard.

## Submission status

**The main method is fixed; the package is not yet ready as an official submission.** A fresh clone can export and check the published CSV, but that is not a reproduction of inference from the raw bundle. The official raw bundle → identical CSV path is not available end-to-end on this branch. The target cache and the wave3/4/5 models and prerequisites are still needed to reproduce the campaign.

The final technical report must follow this method, not Ryu's old write-up or the highest public candidate. Its sources: [wave5 results](../experiments/E06_wave5/RESULTS.md), [plan before training](../experiments/E06_wave5/PLAN.md), and [data/code attribution](../experiments/E05_wave4/reference/SOURCE.md). This document is the method status, not the technical report.

The paired ball-drop stress is slightly worse and the under-count is not solved. The external matches have been exposed historically; there is no evidence of being overfit-free or of superiority on the private GT. History and negative results remain in the [experiment registry](../experiments/REGISTRY.md) and [CAMPAIGNS.md](../experiments/CAMPAIGNS.md).

## Folder layout

- `submission/`: the main choice, the verified CSV export, and readiness status.
- `experiments/E02_wave1/` to `experiments/E07_wave6/`: historical campaigns. Moved from `experiments/exp_farras/` with `git mv`; only path constants were changed (see the [relocation note](../experiments/REGISTRY.md#relocation-note)). The original frozen source remains at the release commit.
- Campaigns `E08_fair_idsse` to `E13_provider_generalization`: benchmarks and further research. A control or primary inside a campaign report applies only to that campaign's protocol and does not replace the submission method.
- `experiments/EXX_*/results/`: aggregate evidence and hashes per campaign. Raw data, caches, models and label/frame-level arrays stay out of Git. The one publication exception is the selected prediction CSV in `submission/artifacts/` (and its copy `submission.csv` at the root).

The runtime source and CSV are on `main` after merging the `farras` history; the original `farras` branch is kept. Other research branches refer to the same decision and report facts without replacing members' research code. The export command never commits or pushes.

## Verification of the decision

The `submission/export.py` path has been run. The official validator accepts 199,319 rows with frames 0..199318; the exported hash is identical to the wave5 candidate. The checks also covered idempotent repeated export, refusal to overwrite a different file, an error when the local CSV is missing, and refusal of a candidate whose hash changed. 61 local links in the index and guide documents also passed.

Structured record: [verification.json](verification.json). This verification did not train or score on public/private GT and does not prove a raw-bundle reproduction.

## Verification of the harness package

The public candidate export also passed on a minimal snapshot containing only the published runtime files, without raw data, tracking caches, weights or GT. The official validator accepts 199,319 rows and the resulting hash is identical to the frozen wave5 candidate. [Publication evidence](publication_verification.json). This is an access/format check of the artifact, not a new score or evidence of raw-bundle inference.
