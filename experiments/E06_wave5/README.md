# E06: Wave5, short-possession loss (FINAL METHOD)

**This is the team's final method**, `farras_wave5_short_gru25_dead45`. Its output is the submission file [`submission.csv`](../../submission.csv) (SHA-256 `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`).

[PLAN.md](PLAN.md) fixed the 96 configurations and the adoption rule before training. The three new models use the wave4 PP-TCN with separate interventions: an auxiliary boundary head, double loss weight for short live spells, or more frequent full ball masking. The decoders compare calibration before/after merging and an H/A reset at dead intervals. The quality mixtures use visibility/entropy, not target labels.

Result: `short_gru25/dead45` passes the adoption rule with external spellF1 0.670099 (+0.013609 over wave4). Public spellF1 stays 0.772414; candidate macroF1 0.902059. Short-spell recall improves, but mean spellF1 on the paired ball-drop stress test drops slightly. Record card: [CARD.md](CARD.md). Negative results, all 96 configurations and reproduction hashes: [RESULTS.md](RESULTS.md). Cross-branch specification: [docs/FINAL_METHOD.md](../../docs/FINAL_METHOD.md).

## Test the result

Levels 1 and 2 need no match data; level 3 needs the frozen caches; level 4 needs the data bundle and a GPU.

### 1. Check the recorded numbers and the submission hash

```bash
python experiments/check_results.py E06
```

Expected: 12 `PASS` lines. These cover selection `short_gru25/dead45`, promoted, 6 eligible configurations, development 0.670099 vs control 0.656490 (gain 0.013609), public 0.772414 / 0.902059, and identical hashes for the public-audit CSV, the independent rebuild, `submission/artifacts/farras_wave5.csv` and the root `submission.csv`.

Then validate and re-export the submission file:

```bash
python evaluation/validate_submission.py submission.csv
python -B submission/export.py --output submission.csv    # prints SHA256 cd59fb90… if unchanged
```

### 2. Run the unit tests

```bash
python -m pip install -r experiments/E06_wave5/requirements.txt
python -B -m pytest -q experiments/E06_wave5
```

Expected: `10 passed` (decoder DP vs exhaustive optimum, dead reset, post-merge calibration, short-spell weights, and more).

### 3. Train from scratch, rebuild the CSV and explain it

Open [`E06_wave5.ipynb`](E06_wave5.ipynb) and run all cells. With `RETRAIN = True` (the default) it trains the final method from scratch through [`train_from_scratch.py`](train_from_scratch.py):

| Section | Step | Needs |
|---|---|---|
| [T1] | download IDSSE (≈ 2.6 GB) + SkillCorner (≈ 1.8 GB); build the training packs and features | internet |
| [T2] | train the set-sequence BiGRU (5 IDSSE matches × 3 rounds, CPU, 1 thread) | CPU |
| [T3] | train the short-loss PP-TCN (19 matches, 3 seeds × 1,500 updates) | GPU recommended |
| [T4] | build the target tracking caches, target pack, features and BiGRU prior from `data/` | the licensed match bundle |
| [1]–[16] | inference, fusion, decoding, CSV, public score, explanations | CPU |

The training part can run on a GPU machine and the target part on the licensed machine:

```bash
python experiments/E06_wave5/train_from_scratch.py public --device cuda   # [T1]–[T3]; no challenge data
python experiments/E06_wave5/train_from_scratch.py target                 # [T4]; needs data/
```

Copy the two checkpoints (`E04_wave3/out/models/hybrid.pt`, `E06_wave5/out/models/short_tcn.pt`) and the 19 training packs from the GPU machine before the target part, then run the notebook with `RETRAIN = False`.

What to expect:
- Section [1] stops with a list of any missing files.
- A retrained checkpoint has a different hash from the frozen one; section [7] then reports how many frames agree with `submission.csv` instead of `reproduced: True`. Different hardware or library versions can change the trained weights slightly.
- Section [8] gives the public spellF1/macroF1 of the retrained model; the frozen submission scores 0.772414 / 0.902059.
- With `RETRAIN = False` and the original frozen files, section [7] must print `reproduced: True` (hash `cd59fb90…`).
- Sections [9]–[16] explain the predictions of whichever model was run.

### 4. Reproduce the whole campaign

From the repository root, Python 3.10. The cache/data/features and the frozen wave3/4 results must be available. This does not change old artifacts or download new datasets. SkillCorner MIT, IDSSE CC BY 4.0; source/data credit in [E05_wave4/reference/SOURCE.md](../E05_wave4/reference/SOURCE.md). Metrica is evaluation only; the wave5 families use no anchor tree.

```bash
export FARRAS_SOURCE_EXPERIMENTS=/path/to/workspace/experiments   # parent folder of _cache
export FARRAS_TRAIN_DEVICE=cuda
python -m pip install -r experiments/E06_wave5/requirements.txt
python -B experiments/E06_wave5/test_invariants.py
python -B experiments/E06_wave5/run.py dev
python -B experiments/E06_wave5/prepare_stress.py
python -B experiments/E06_wave5/prepare_reference.py
python -B experiments/E06_wave5/run.py holdout
```

`python -B experiments/E06_wave5/run.py public` is the campaign's post-freeze diagnostic audit. It is not part of the export or required to run the main method. The recorded public audit does not need to be repeated.

`prepare_reference.py` uses CPU in a separate process, so the old GRU does not trigger the local CUDA shutdown crash and the checkpoint module `models` does not clash with the TCN. Do not import the GRU and TCN models under the same module name in one process. The CUDA TCN must pass an actual fitting/inference smoke test; the backend is hashed in the registry.

What to compare after a rerun:
- `out/selection.json` hashes to `6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`.
- `out/candidates/selected.csv` hashes to `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.
- The holdout and short-spell tables match [RESULTS.md](RESULTS.md).

## Protocol

- Training on the 19 matches from wave4, 6 grouped folds, 3 seeds per model, 1,500 steps per seed. The three new models give 21 ensemble fits / 63 seed fits, not 96 independent models.
- 96 probability-family/decoder configurations produce 2,208 evaluations on 23 development scenarios. Provider-balanced mean spellF1 is the objective; short GT is not dropped.
- A replacement for the wave4 control must gain at least 0.01, win at least 2/3 providers, with no provider dropping more than 0.03. If none qualifies, the control is kept; the best raw alternative is still reported.
- The old holdout is already exposed. The 6 paired SkillCorner ball-drop stress cases were made after the freeze; they are not 6 new matches. The 16 holdout scenarios come from 9 unique matches.
- Only four fixed roles were scored on public after refit/rebuild: selected, best_new, previous_wave4, pure_boundary. Roles with the same hash are not independent results. No 96-way public sweep.

Source, recipe and inputs were hashed before training; selection and checkpoint hashes were frozen before holdout/stress/public. An algorithm change needs a new campaign, not a hash replacement. `dev` can reuse identical finished caches but does not overwrite the frozen selection.

## Artifacts

- `cache/`: predictions, stress packs/features and frozen GRU stress probabilities (ignored).
- `out/`: pre-registration, ranking, model checkpoints, holdout/short-duration diagnostics, public audit and submission CSVs (ignored).
- `out/notebook/`: written by `E06_wave5.ipynb` (ignored).
- `results/wave5_*.json`: aggregates safe to share (committed).
- [RESULTS.md](RESULTS.md): explanation of the results.

 Other members' source and experiments stay separate.
