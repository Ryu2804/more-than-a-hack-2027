# E02: Wave1, rule and learned families on Metrica

The first Farras campaign: 8 method families (hysteresis, Viterbi, duration, persistence, logistic, histogram boosting, extra trees, temporal boosting; see [`methods/`](methods/)), 24 configurations and 96 external evaluations on Metrica sample Games 1/2. Each game is used in its original condition and with synthetic occlusion, in two leave-one-game-out folds. Game 3 was held out until after the freeze.

Result: selected `m01_hysteresis/r15` (radius 1.5 m, confirm 0.4 s, gap→dead 8 s), external mean spellF1 0.460517, Game 3 0.474935 (original) / 0.374534 (occluded), public diagnostic 0.442105. It was superseded by E03. Record card: [CARD.md](CARD.md). Detailed results (shared with wave2): [E03_wave2/RESULTS.md](../E03_wave2/RESULTS.md). Protocol and inputs: [CAMPAIGNS.md](../CAMPAIGNS.md).

## Test the result (no match data needed)

```bash
python experiments/check_results.py E02
```

Expected: `PASS` for the selected `m01_hysteresis/r15` and external mean spellF1 0.460517 in `results/wave1_metrics.json`. This campaign has no unit tests of its own.

## Reproduce the campaign (needs the Ryu tracking cache and Metrica sample data)

From the repository root, Python 3.10:

```bash
python -m pip install -r experiments/E02_wave1/requirements.txt
python -B experiments/E02_wave1/data.py                    # prepares the cached inputs
python -B experiments/E02_wave1/run.py all --workers 4     # 24 configurations, external selection
python -B experiments/E02_wave1/run.py public              # post-freeze public diagnostic only
```

Inputs are read from `experiments/_cache/` (`atd_full.npy`, `team_dist.npy`, `ext_metrica/…`); set `FARRAS_SOURCE_EXPERIMENTS` to the parent folder of `_cache` if it lives elsewhere ([CAMPAIGNS.md](../CAMPAIGNS.md)). What to compare: the selection must be `m01_hysteresis/r15`, every configuration's external mean must match `results/wave1_metrics.json`, and the candidate CSV must hash to `6a022089301a08d9d469fbc4630cb65c3f4a34b1d4de1193488422370e63b52d`. The relocation rerun recorded in `../E03_wave2/results/relocation_verification.json` reproduced all of these.
