# E01: Official baseline (nearest player + majority vote)

[`E01_baseline.ipynb`](E01_baseline.ipynb) runs the organisers' baseline: the nearest tracked player to the ball gives a team code per frame, followed by a majority vote over ±0, ±25, ±50 or ±125 frames. Only the window changes, which shows how differently spellF1 and macroF1 react. Record card: [CARD.md](CARD.md).

## Test the result

### 1. Without match data

The four baseline CSVs are committed in `evaluation/results/`, so they can be scored directly:

```bash
python experiments/check_results.py E01      # public spellF1 of the four baselines
python evaluation/run_evaluation.py          # official scoreboard on the public 5-minute GT
```

Expected public spellF1: raw 0.037838, smooth25 0.286792, smooth50 0.325581, smooth125 0.424242. The scoreboard ranks smooth125 first on the public window. The full match ranks smooth50 first (published full-match spellF1 0.2433, see `EVALUATION.md`). That difference is the reason no window is chosen from the public window.

### 2. With the data bundle

1. Unpack the data into `data/` ([DATA.md](../../DATA.md)).
2. Regenerate the baselines and check they are unchanged:

   ```bash
   python starter/baseline.py
   git status --short evaluation/results/      # expect no changes
   ```

3. Rerun the notebook (writes its own copies to `experiments/E01_baseline/out/`, ignored):

   ```bash
   jupyter nbconvert --to notebook --execute --inplace experiments/E01_baseline/E01_baseline.ipynb
   ```

   Its scores on the public GT must equal those of step 1.
