# E04: Wave3, identity-free set+BiGRU combined with ball-control features

The design is frozen in [PLAN.md](PLAN.md). Jeremy's reference is pinned at commit `0e35aae54e09ca744441681fb076f656bbc88b5a`; the parser/degrader source is kept verbatim and attributed in [reference/SOURCE.md](reference/SOURCE.md).

This campaign is a **controlled adaptation**, not a byte-identical reproduction of Jeremy's model: a smaller model, 8 training epochs, tokens from 5-fps observations, five development matches and two IDSSE holdout matches. Jeremy's 0.581818 and Farras's 0.56 diagnostic are only historical public numbers, not tuning targets.

Result: selected candidate `hybrid_tree25/centered2`, development spellF1 0.615394, public diagnostic 0.575610; an independent refit reproduces all 199,319 target rows. Its BiGRU is reused, frozen, by the final method (E05, E06). Record card: [CARD.md](CARD.md). Full results and the ranking of all 66 configurations: [RESULTS.md](RESULTS.md). Audit of Jeremy's branch: [JEREMY_REVIEW.md](JEREMY_REVIEW.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers.** Recomputes the headline values in [CARD.md](CARD.md) from `results/wave3_*.json`:

   ```bash
   python experiments/check_results.py E04
   ```

   Expected: `PASS` for the selected family/decoder `hybrid_tree25/centered2`, best development spellF1 0.615394, 66 configurations and public diagnostic 0.575610.

2. **Run the unit tests** (decoders, invariances, feature preservation). They use synthetic inputs only:

   ```bash
   python -m pip install -r experiments/E04_wave3/requirements.txt
   python -B -m pytest -q experiments/E04_wave3
   ```

   Expected: `6 passed`. The tests import the vendored Jeremy parser, which needs `lxml`.

## Reproduce the campaign (needs data and caches)

Run from the repository root with Python 3.10 and `requirements.txt`. The installed runtime is PyTorch 2.13.0+cu126, but the campaign uses single-thread CPU: the local cuDNN GRU backend crashed at process shutdown. CPU and native CUDA passed the lifecycle smoke test; CPU was faster than native CUDA on the same smoke test. Models and epochs were not reduced. No external training service was used.

```bash
export FARRAS_SOURCE_EXPERIMENTS=/path/to/workspace/experiments   # parent folder of _cache
export FARRAS_TRAIN_DEVICE=cpu
python -B experiments/E04_wave3/prepare_idsse.py
python -B experiments/E04_wave3/prepare_local.py
python -B experiments/E04_wave3/test_invariants.py
python -B experiments/E04_wave3/run.py dev
python -B experiments/E04_wave3/run.py holdout
python -B experiments/E04_wave3/run.py public
```

`prepare_idsse` downloads the metadata/tracking of seven public matches (about 2.62 GB in total) from the [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data); sizes and LFS SHA-256 are verified. IDSSE is CC BY 4.0, Bassek et al. 2025, [paper](https://doi.org/10.1038/s41597-025-04505-y), [data](https://doi.org/10.6084/m9.figshare.28196177). Downloads and derived data stay local (gitignored).

`prepare_local` needs the same Ryu and Metrica caches as wave1/2. Run the wave1 and wave2 preparation first if those caches are not available yet; see [CAMPAIGNS.md](../CAMPAIGNS.md). The override points at the parent folder of `_cache`. No raw target GT is needed. The target team mapping is recovered from the nearest-distance cache only when the evidence meets the thresholds and the reconstructed minima match; it does not claim to read XML metadata that is not available.

What to compare after a rerun: `out/selection.json` must select `hybrid_tree25/centered2`; `out/holdout_report.json` must reproduce the target CSV SHA-256 `337734b6bf9b70bda6c9a16b549e39e3c90bdbd53fc6073c140ab07cd06b6096`; the development ranking must match the table in [RESULTS.md](RESULTS.md).

## Selection and artifacts

- `cache/data/`: observations, features and proxy labels; gitignored.
- `cache/jeremy/`: branch audit snapshot; gitignored.
- `cache/predictions/`: fold probabilities; gitignored.
- `out/preregistered.json`: configurations and code/input hashes before training.
- `out/models/`: local fitted checkpoints.
- `out/selection.json`, `external_scoreboard.csv`: ranking of the 66 configurations, based on external development only.
- `out/holdout_report.json`: four pre-specified candidates, four holdout scenarios, and an independent reproduction of the target candidate.
- `out/candidates/*.csv`: four local target CSVs; gitignored.
- `out/public_audit.json`: post-freeze diagnostic for the four candidates only.
- `results/wave3_*.json`: committed aggregate evidence.

Training is serialised on single-thread CPU. Workers only implement components; the harness runs the fits and prediction checks. `dev` can resume finished caches with the same pre-registration hash. After the selection is frozen, do not rerun the campaign with changed code or overwrite old evidence.

## Limits

IDSSE proxies have a different dead-ball definition and coverage from the challenge. Metrica uses event-derived labels. The two IDSSE holdout matches were not used for training in this campaign, but Jeremy used them and their aggregate results influenced the design. Metrica 3 had been evaluated in wave2. The public diagnostic does not prove a private/full-match score.

Offline predictions use two-way context and gap remaining. The 5-fps dataset lowers the resolution of possession boundaries. The cost/architecture/configuration grid was not extended after the scores were known. Negative results and possession-count errors are reported.
