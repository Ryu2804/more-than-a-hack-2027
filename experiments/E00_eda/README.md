# E00: Exploratory data analysis

[`E00_eda.ipynb`](E00_eda.ipynb) describes the target match before any modelling. It covers:
- graded frame blocks
- where and for how long the ball disappears from tracking
- player-track quality (fragmentation, identity jumps)
- ball–player geometry
- team shape and direction of attack
- the content and reliability of Smart Tagging
- broadcast homography

The public ground truth is used only to check alignment, never to tune anything. Record card with the headline numbers: [CARD.md](CARD.md).

## Test the result

### 1. Without match data

```bash
python experiments/check_results.py E00
```

This checks the public-GT facts the notebook reports from the committed `evaluation/gt/` files. Expected `PASS` lines: 7,500 graded frames, 17 possessions, and 3,454 H, 2,577 A and 1,469 D frames.

### 2. With the data bundle (full rerun)

1. Unpack the match bundle into `data/` as described in [DATA.md](../../DATA.md) (or set `MTAH_DATA_DIR`).
2. Install Jupyter in your environment (`pip install jupyter`) and run from anywhere inside the repository:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace experiments/E00_eda/E00_eda.ipynb
   ```

   The first run parses the full ATD (about a minute) and caches it in `experiments/_cache/`.
3. Compare the printed values with the original run (also quoted in Table I of [`carried_by_admin.pdf`](../../carried_by_admin.pdf)):

| Cell output | Expected |
|---|---|
| frames / graded | 199,319 / 158,857 |
| home / away slots | 24 / 22 |
| ball visible (all / graded) | 56.9% / 59.3% |
| tracked players per graded frame | mean 20.5, max 25 |
| ball-missing gaps | 150; median 10.7 s, P90 33.8 s, max 221 s |
| gaps < 1 s / > 10 s; missing frames in gaps > 10 s | 11.3% / 52.7%; 85.2% |
| P(ball missing 1 s later \| visible) | 3.6% |
| tracklets | 2,218; median 13.6 s; P90 201 s |
| jumps > 2 m on graded frames | 0.52 per 1000 frames |
| nearest-player distance; margin < 2 m | median 1.9 m (P90 7.0 m); 32.5% |
| Smart Tagging team agreement (raw / after flip) | 53.7% / 63.4% |
| broadcast homography valid; invalid stretches | 65.8%; 123 |

The original run with outputs, from before the English translation, is at commit `efdb82e` as `experiments/00_eda.ipynb`.
