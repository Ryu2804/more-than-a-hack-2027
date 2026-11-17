# E05: Wave4, Ryu's PP-TCN trained across providers

[PLAN.md](PLAN.md) locks the split, the 64 configurations, the comparison limits and the selection rule before training. Ryu's source is pinned at commit `5eaf3e595e91fa0502aac48037d0cae8c356ec6f`; [reference/SOURCE.md](reference/SOURCE.md) holds credit and provenance.

Ryu's latest report mentions a public spellF1 of about 0.747 for the player-set TCN, while the old technical document still says 0.422. Do not mix those versions, or assume the no-Metrica checkpoint / cached-input adapter gives the same number.

Measured result: the externally selected candidate `pp_gru25/no_shift` has development spellF1 0.656490 and **public spellF1 0.772414**, the same as the pure PP-TCN. Ryu's shipped adapter 0.760000; frozen wave3 0.575610. The 6 SkillCorner holdout matches improve, but Metrica 3 is lower than wave3. It became the control of E06. Record card: [CARD.md](CARD.md). All results, comparison limits and refit hashes: [RESULTS.md](RESULTS.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/wave4_*.json`:

   ```bash
   python experiments/check_results.py E05
   ```

   Expected: `PASS` for `pp_gru25/no_shift`, development equal-provider spellF1 0.656490, 64 configurations, public diagnostic 0.772414.

2. **Run the unit tests** (decoder DP, gates, fusion, player-set permutation, masking, and the guard that refuses the shipped reference on in-sample providers):

   ```bash
   python -m pip install -r experiments/E05_wave4/requirements.txt
   python -B -m pytest -q experiments/E05_wave4
   ```

   Expected: `11 passed`.

## Reproduce the campaign (needs data and caches)

From the repository root, Python 3.10 with the wave3 dependencies, CUDA Conv1d if the runtime smoke test passes. The wave3 Metrica/target/IDSSE caches are required; raw SkillCorner data is downloaded from the public MIT-licensed upstream. No data upload or remote training.

```bash
export FARRAS_SOURCE_EXPERIMENTS=/path/to/workspace/experiments   # parent folder of _cache
export FARRAS_TRAIN_DEVICE=cuda
python -m pip install -r experiments/E05_wave4/requirements.txt
python -B experiments/E05_wave4/prepare_skillcorner.py
python -B experiments/E05_wave4/prepare_features.py
python -B experiments/E05_wave4/prepare_prior.py dev
python -B experiments/E05_wave4/test_invariants.py
python -B experiments/E05_wave4/run.py dev
python -B experiments/E05_wave4/prepare_prior.py holdout
python -B experiments/E05_wave4/run.py holdout
python -B experiments/E05_wave4/run.py public
```

`prepare_prior` runs on CPU in a separate process, because the wave3 checkpoint uses its own `models` module and the local CUDA GRU backend fails at shutdown. The TCN campaign uses CUDA only after an actual Conv1d+PP fitting/refit/checkpoint smoke test finishes cleanly. `FARRAS_TRAIN_DEVICE=cpu` can be chosen explicitly before pre-registration; after the freeze the backend must match the registry.

Do not rerun `prepare_prior dev` after pre-registration, because its NPZ inputs are hashed. `dev` can resume finished predictions/checkpoints with identical source/inputs but refuses to overwrite the frozen selection. For an algorithm change, start a new campaign instead of replacing old hashes.

What to compare after a rerun: `out/selection.json` must select `pp_gru25/no_shift`; the target CSV must hash to `ddd75416e7929183954d27191b11ddab1ac049fc0e703f802db2209884635e9a`; the frozen selection must hash to `b5269672e86cce891c7ce28054ce17a6acfa4c4979e11236757212d908acdd05`; the development table must match [RESULTS.md](RESULTS.md).

## Data and validation

New training uses only 19 matches: 5 IDSSE + 14 SkillCorner. IDSSE uses the three wave3 corruption rounds; SkillCorner uses native broadcast tracking with stochastic masking during training. Six grouped folds exclude every round of the same match. Metrica 1/2 is zero-shot stress validation and does not train the new TCN.

Campaign holdout: 2 IDSSE + 6 SkillCorner + Metrica 3 original/degraded. IDSSE/Metrica 3 were seen in earlier campaigns; the 6 SkillCorner matches are new for Farras but not for Ryu's research. Ryu's shipped weights were trained on all 27 IDSSE/SkillCorner matches and are therefore not scored as held-out on those matches. The shipped-reference adapter covers only Metrica 3 and the target.

The 3-seed weights of each TCN ensemble and the GRU/tree priors feed 8 probability families. Eight fixed decoders give 64 configurations. The main metric is mean spellF1 with equal provider weights; macroF1 is the tie-breaker. Only four fixed candidates were scored on public after the freeze; there was no 64-way public search.

## Artifacts

- `cache/raw/`: local raw SkillCorner; gitignored.
- `cache/data/`, `cache/features/`, `cache/prior/`, `cache/predictions/`: prepared data and probabilities; gitignored. `cache/features/target_clean.npz` and `cache/prior/target_clean.npz` are inputs of the final method (E06).
- `cache/ryu_weights/`: Ryu's source checkpoints; gitignored.
- `out/preregistered.json`, `out/selection.json`, `out/external_scoreboard.csv`: registry/freeze of the 64 configurations.
- `out/models/`: the 3 local fitted TCN ensembles.
- `out/holdout_report.json`: fixed-reference evaluation, Ryu-source adaptation and independent-refit hash.
- `out/public_audit.json`: four public diagnostics, never a basis for replacing the candidate.
- `out/candidates/*.csv`: per-frame labels, kept local.
- `results/wave4_*.json`: committed aggregate evidence.

The wave1/2/3 code and other members' branches are unchanged. The original source, provider data and raw caches keep their own provenance. The selected PP-TCN and GRU are not trained on Metrica; families using the old anchor tree still inherit its Metrica training and the unclear license. Metrica is also used for zero-shot evaluation.
