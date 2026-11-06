---
title: Possession DP Pipeline v2 - Gated Dead Evidence and GT-Free Lambda
type: experiment-report
tags: [experiment, possession, dp, dead-ball, spellf1, public-window, synthetic]
status: inconclusive
last_updated: 2026-11-05
---

# Possession DP Pipeline v2

Folder: `experiments/002_possession_dp/` (this file sits next to the code it describes).
Script: `experiments/002_possession_dp/run.py`
Pre-registration: written before the main run; the summary, the one amendment, and the reasoning are in `NOTES.md`. Raw numbers are in `results/results.json`.

## 1. Status

**Inconclusive.** Evidence level: public-window diagnostic + synthetic mechanism. No full-match
ground truth is available to us, so nothing here is a full-match result and nothing here is used to
pick a parameter from the public window.

Sub-verdicts, kept separate because they do not share the same status:

| Claim | Verdict | Basis |
|---|---|---|
| Pre-registered `lambda = 4` works on real tracking | **Not supported** | 95 predicted possessions in the 5-minute window (GT 17); 1,356 over the full match vs count band [110.4, 662.4] |
| Gated observed-only dead evidence adds spellF1 over plain smoothing | **Inconclusive** | +0.098 over `baseline_smooth125`, paired CI90 [-0.015, +0.227] includes 0; synthetic gain is tautological by construction |
| Missing-aware DP adds value over the same evidence as a hard mask | **Not supported** | `dpL_final` -0.036 vs `smooth125_dead_final`, paired CI90 [-0.161, +0.061] |
| DP with gated evidence beats `baseline_smooth50` | Diagnostic only, not a claim | +0.161, paired CI90 [+0.064, +0.262]; public window may not be used for selection |
| GT-free count rule selects a usable lambda | **Supported as a rule, not as accuracy** | `lambda = 32` gives 280 full-match possessions, band centre 270.4 |

The public window was already inspected in the v1 smoke run on 2026-11-05. Under RULES section 2,
EVALUATION.md ("never to choose between two versions of it, and never as training data") and
DATA.md ("not for ranking, not training data and not a tuning set"), every public-window number
below is a correctness diagnostic, not a selection signal.

## 2. Pertanyaan

Does a missing-aware 3-state DP (H/A/D) with **gated, observed-only dead-ball evidence**
(kinematics, pitch geometry, Smart Tagging restarts, gated long-missing runs) produce better
spellF1 than the strongest smoothing baseline (`baseline_smooth125`) and than the same evidence
used as a hard mask on top of smoothing?

Secondary: is the pre-registered synthetic default `lambda = 4` transferable to real tracking, and
can lambda be chosen without ground truth?

## 3. Data

| Item | Value |
|---|---|
| Match | FC Barcelona (H) vs Athletic Club (A), LaLiga 2024-25 md2 |
| Tactical tracking | `data/tactical/BAR-ATH_tactical_atd.txt` + `_atd.xml` |
| Frames | 199,319 (25 fps) |
| Graded frames | 158,857 (1st half 11,185-90,651; 2nd half 110,808-190,197) |
| Ball present | 113,488 frames (56.9% of all) |
| Smart Tagging restarts | 59 SET PIECES instances |
| Public GT | `evaluation/gt/BAR-ATH_possession_gt_public.csv`, 7,500 graded frames (11,185-18,684), labels H 3,454 / A 2,577 / D 1,469, **17 possessions**, mean 14.19 s |
| Video / bounding boxes | **absent from the bundle** |

Input hashes (sha256, recorded in `results.json`):

| File | sha256 |
|---|---|
| `BAR-ATH_tactical_atd.txt` | `f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431` |
| `BAR-ATH_tactical_atd.xml` | `327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a` |
| `BAR-ATH_tactical_smart_tagging.xml` | `0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019` |
| public GT CSV | `e596a9af9626c623bb94be6fd350e12ee9a75c76a696cfadca3bc01735e92487` |

No match data, video, or GT is copied into this folder. Submissions here are derived predictions
and can be regenerated from the script.

## 4. Metode

### 4.1 Decision layer

Missing-aware Viterbi DP over 3 states H/A/D, run per half, with switch penalty `lambda`:

- observed ball assigned to nearest player: cost 0 for that team, 1 for the other, 1 for D;
- ball missing: cost 0 for all three states (missing never forces D);
- dead evidence true: cost 0 for D, 1 for H and A;
- frames outside the two halves forced to D.

`baseline_raw` is nearest player with D where the ball is missing; `baseline_smooth*` is the
starter majority-vote smoother (`baseline.py`); `smooth125_dead_*` applies the same dead mask as a
hard post-process to `smooth125`.

### 4.2 Evidence components (graded frames only)

| Mask | Definition | Graded frames |
|---|---|---|
| `low_speed` | Savitzky-Golay trend speed < 1.0 m/s for >= 30 frames | 6,553 (4.1%) |
| `stationary` | net displacement < 1.5 m over 1 s, presence density >= 0.8 | 4,846 (3.1%) |
| `kin` | `low_speed` OR `stationary` | 7,665 (4.8%) |
| `geom` | ball > 0.5 m outside pitch for >= 15 frames | 3,164 (2.0%) |
| `set_piece` | first 5 s (125 frames) of each Smart Tagging SET PIECES instance | 7,287 (4.6%) |
| `long_missing` | ball missing >= 5 s continuous (v1 component, ungated) | 64,446 (40.6%) |
| `miss_gated` | `long_missing` AND (restart within +/-75 frames OR last/first ball position within 3 m of the touchline) | 44,902 (28.3%) |
| `all` | v1 union: `long_missing` OR `low_speed` OR `set_piece` | 73,137 (46.0%) |
| `final` | `kin` OR `geom` OR `set_piece` OR `miss_gated` | 56,558 (35.6%) |

`PREREG_AMENDMENT.md` records the only change made after the first test run: the missing gate was
tightened from 8 m + a 15 s exemption to 3 m with no exemption, because 56 of 132 long missing runs
exceeded 15 s and the exemption covered 75% of the ungated mask, making the gate inert (60,028 vs
64,446 frames). No lambda change accompanied it, and no public-window number was used to make it.

### 4.3 Lambda

`lambda = 4` was pre-registered from the synthetic note. Because pre-registration also requires a
ground-truth-free selection rule, the full match was swept over `{4, 8, 16, 25, 32, 50, 64}` with
the `final` evidence and the arm chosen by the accepted count-band rule: predicted possession count
inside [110.4, 662.4], closest to centre 270.4. This uses **no ground truth**.

| lambda | 4 | 8 | 16 | 25 | 32 | 50 | 64 |
|---|---|---|---|---|---|---|---|
| full-match possessions | 1356 | 836 | 483 | 339 | **280** | 191 | 146 |
| mean duration (s) | 3.0 | 4.9 | 8.4 | 12.0 | 14.6 | 21.4 | 27.5 |
| D fraction (graded) | 0.356 | 0.357 | 0.358 | 0.358 | 0.356 | 0.356 | 0.368 |

Selected: `lambda = 32` (280 possessions, distance 9.6 from centre, inside band).

### 4.4 Arms

`baseline_raw`, `baseline_smooth50`, `baseline_smooth125`, `smooth125_dead_long`,
`smooth125_dead_final`, `dp4_none`, `dp4_final`, `dpL_none`, `dpL_long_missing`, `dpL_final`
(L = 32), and negative controls `negative_constant_D`, `negative_constant_H`.

Synthetic harnesses: (A) i.i.d. observation noise on an independent truth; (B) autocorrelated
confusion episodes (`p_start = 0.01`, mean episode 2 s) with truth aligned to the `final` evidence
mask, 3 seeds. Neither harness is calibrated to the real error rate; harness B is milder than the
real 28.4 nearest-player switches/minute, and its dead states sit on the same mask the DP receives,
so evidence gains there are an upper bound by construction.

Bootstrap: 1000 draws, seed 0, 5th-95th percentiles, block resampling over GT possessions; paired
differences use the same resampling for both arms. `n_pred` is held fixed per arm, so intervals for
count-changing arms are optimistic.

## 5. Hasil

### 5.1 Public window (diagnostic only, 17 GT possessions, 7,500 graded frames)

| Arm | spellF1 | CI90 | macroF1 | spells | dur (s) | frag | D F1 |
|---|---|---|---|---|---|---|---|
| baseline_raw | 0.0378 | 0.0189-0.0595 | 0.5972 | 131 | 2.02 | 6.52 | 0.420 |
| baseline_smooth50 | 0.3256 | 0.2140-0.4465 | 0.6207 | 26 | 10.80 | 1.43 | 0.323 |
| baseline_smooth125 | 0.4242 | 0.2667-0.5939 | 0.5812 | 16 | 18.46 | 0.81 | 0.146 |
| smooth125_dead_long | 0.4242 | 0.2788-0.5818 | 0.6740 | 16 | 16.55 | 0.95 | 0.421 |
| smooth125_dead_final | 0.5222 | 0.3667-0.6778 | 0.6850 | 19 | 13.60 | 1.29 | 0.455 |
| dp4_none | 0.0561 | 0.0299-0.0860 | 0.4474 | 90 | 3.33 | 4.29 | 0.000 |
| dp4_final | 0.0786 | 0.0393-0.1286 | 0.6134 | 95 | 2.72 | 4.90 | 0.455 |
| dpL_none | 0.3742 | 0.2194-0.5419 | 0.5061 | 14 | 21.43 | 0.67 | 0.000 |
| dpL_long_missing | 0.4000 | 0.2424-0.5576 | 0.6686 | 16 | 16.55 | 0.95 | 0.421 |
| dpL_final | 0.4875 | 0.3250-0.6500 | 0.6803 | 15 | 17.45 | 1.00 | 0.438 |
| negative_constant_D | 0.0000 | 0.0000-0.0000 | 0.1092 | 0 | 0.00 | 0.05 | 0.328 |
| negative_constant_H | 0.0000 | 0.0000-0.0000 | 0.2102 | 1 | 300.00 | 0.05 | 0.000 |

Paired bootstrap differences:

| Comparison | delta | CI90 | draws positive |
|---|---|---|---|
| dpL_final - baseline_smooth125 | +0.0630 | -0.0341 to +0.1705 | 83.5% |
| dpL_final - baseline_smooth50 | +0.1607 | +0.0642 to +0.2616 | 99.8% |
| dpL_final - smooth125_dead_final | -0.0358 | -0.1611 to +0.0611 | 33.4% |
| dpL_final - dpL_none | +0.1125 | +0.0008 to +0.2403 | 95.3% |
| dpL_final - dp4_final | +0.4079 | +0.2554 to +0.5643 | 100% |
| smooth125_dead_final - baseline_smooth125 | +0.0988 | -0.0152 to +0.2273 | 91.2% |

### 5.2 Synthetic harness A (i.i.d. noise, 3 seeds, mean spellF1)

| Arm | mean | min-max |
|---|---|---|
| baseline_smooth50 | 0.171 | 0.166-0.179 |
| baseline_smooth125 | 0.230 | 0.214-0.258 |
| dp4_none | 0.531 | 0.527-0.537 |
| dp16_none | 0.527 | 0.519-0.536 |
| dp32_none | 0.505 | 0.502-0.508 |
| dp50_none | 0.464 | 0.453-0.480 |

Under i.i.d. noise the DP is nearly lambda-insensitive; this is the regime that produced the
synthetic note's `lambda = 4` default.

### 5.3 Synthetic harness B (autocorrelated confusion, 3 seeds, mean spellF1)

| Arm | mean | min-max |
|---|---|---|
| baseline_smooth50 | 0.363 | 0.349-0.374 |
| baseline_smooth125 | 0.342 | 0.329-0.352 |
| smooth125_dead_final | 0.555 | 0.538-0.574 |
| dp4_final | 0.433 | 0.421-0.442 |
| dp8_final | 0.473 | 0.463-0.480 |
| dp16_final | 0.540 | 0.524-0.550 |
| dp25_final | 0.567 | 0.555-0.577 |
| dp32_final | 0.578 | 0.575-0.584 |
| dp50_final | 0.558 | 0.544-0.574 |
| dp64_final | 0.511 | 0.500-0.526 |

Paired screen on harness B (pre-registered rule: delta >= 0.04, CI excludes 0 in >= 2 of 3 seeds,
positive in >= 2 of 3):

| Comparison | mean delta | seeds CI excludes 0 | active |
|---|---|---|---|
| dp32_final - baseline_smooth125 | +0.236 | 3/3 | yes |
| smooth125_dead_final - baseline_smooth125 | +0.213 | 3/3 | yes |
| dp32_final - dp4_final | +0.145 | 3/3 | yes |
| dp4_final - baseline_smooth125 | +0.091 | 3/3 | yes |
| dp32_final - smooth125_dead_final | +0.023 | 1/3 | no |

The lambda optimum moves from 4 under i.i.d. noise to 25-32 under autocorrelated confusion, which is
the mechanism behind the real-data failure of `lambda = 4`. The harness cannot fix the value: its
switch rate is 11.5-12.8/min versus 28.4/min on the real tracking.

### 5.4 Full-match GT-free diagnostics

| Arm | predicted possessions | mean duration (s) | inside count band |
|---|---|---|---|
| baseline_raw | 1949 | 1.9 | no |
| baseline_smooth50 | 594 | 7.2 | yes |
| baseline_smooth125 | 398 | 12.4 | yes |
| smooth125_dead_long | 306 | 12.3 | yes |
| smooth125_dead_final | 415 | 9.1 | yes |
| dp4_none | 1289 | 4.9 | no |
| dp4_final | 1356 | 3.0 | no |
| dpL_none | 225 | 28.2 | yes |
| dpL_long_missing | 311 | 12.0 | yes |
| dpL_final | 280 | 14.6 | yes |
| negative_constant_D | 0 | 0.0 | no |
| negative_constant_H | 2 | 3177.1 | no |

Self-supervised masking probe (30 windows of 125 frames with the ball visible, observations hidden,
mean fraction of labels that change): `dpL_final` 0.259, `dp_none` 0.243, missing-to-D control
0.781. Structural only: the missing-aware DP holds its state through an occlusion while the naive
missing-to-D rule rewrites the window.

Official scorer parity: max absolute difference 4.7e-07 across all 12 arms (same metric module, so
this verifies file and pipeline consistency, not independent validation). Scoreboard order matches
the internal table.

## 6. Interpretasi

**Main effect is the evidence mask, not the DP.** In the public window, adding the `final` mask to
`smooth125` moves spellF1 0.4242 -> 0.5222 and D F1 0.146 -> 0.455; the DP with the same mask lands
at 0.4875 and is not separated from the hard-mask version (delta -0.036, CI crosses 0). So the
decision layer earns nothing we can demonstrate on this window.

**Pre-registered lambda does not transfer.** `lambda = 4` assumes i.i.d. label noise; the real
nearest-player sequence switches 28.4 times per minute, so at lambda 4 the DP follows noise and
fragments into 95 possessions in 5 minutes. This is a calibration failure of the accepted
synthetic note, not a rescaling detail: at lambda 32 the same code produces 15 possessions and
0.4875 spellF1. The synthetic curve reproduces the direction (optimum shifts to 25-32) but cannot
supply the number.

**The GT-free count rule is coherent but coarse.** It selects lambda 32 and its full-match count
(280) is closer to the band centre than any baseline. However, seven arms sit inside the band, and
`smooth125_dead_final` at 415 is inside it too, so the rule can rank but cannot discriminate. It
was pre-registered and is not a substitute for full-match accuracy.

**Why no improvement claim.** The public window holds 17 possessions, giving a CI width around 0.3;
every difference that matters here is inside one CI. EVALUATION.md says overlapping intervals mean
the gap is not established, and the window is forbidden as a selection set anyway.

Failure modes observed:

- DP without dead evidence empties the D class entirely (D F1 0.000); the metric still rewards it
  because D is only ~20% of graded frames and possessions are unaffected, which is a trap.
- Ungated `long_missing` marks 40.6% of graded frames dead, conflating occlusion with stoppage;
  `dpL_long_missing` equals `smooth125_dead_long` exactly, i.e. gating is what changes behaviour.
- Team identity switches are not repaired; the ATD team label is used as-is, and the 28/min switch
  rate is the dominant noise source.
- Ball speed is provided on only ~9% of present-ball frames; the fallback derivative is now divided
  by the real frame gap and refuses gaps > 25 frames, but it is still a coarse estimator.
- The 5 s set-piece window is a constant prior, not measured dead time.
- The synthetic evidence harness places truth on the evidence mask, so its evidence gains are
  tautological; only the lambda direction is informative.
- Negative controls score 0.0000 as expected, so the metric cannot be gamed by a constant label.

ByteTrack / BoT-SORT: **not run**. The bundle contains point tracking and metadata only; no video
and no bounding-box detections. Those trackers need boxes plus detector confidences, so applying
them here is not possible, and applying them to a single tracked point would be meaningless.

## 7. Compute

| Item | Value |
|---|---|
| Wall clock, full run | 51.1 s |
| User CPU / system CPU | 51.3 s / 0.38 s |
| Peak RSS | 171 MB |
| GPU | none used; CPU only, no GPU needed |
| Data load | 2.4 s |
| Evidence build | 0.10 s |
| One DP pass over 199,319 frames | 0.38 s |
| 7-lambda full-match sweep | ~3 s |
| Mask probe (90 DP passes) | ~34 s |
| Hardware | local Linux x86_64, Python 3.12.3 |

## 8. Reproducibility

Command (run from `more-than-a-hack-2027/`):

```
.venv/bin/python experiments/002_possession_dp/run.py \
  --data-dir ../data --bootstrap 1000 --seed 0 --synthetic-seeds 3 --write-submissions
```

Then, already executed by the script and reproducible separately:

```
.venv/bin/python evaluation/run_evaluation.py \
  --results experiments/002_possession_dp/out/submissions \
  --out experiments/002_possession_dp/out/official_reports --bootstrap 1000
```

- Script sha256 (the committed `run.py`): `28f664eba9699e078eb1aee56ace2a6d07156193e70689699325070d8fce87cc`
- Repo commit: `f8f2629ff1e0985f81d7e59d8413678514ee1530` (`experiments/` is untracked; the script
  hash is the provenance until it is committed)
- Versions: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1
- Determinism: three complete runs with the same command produced identical metrics for all 12
  public arms, identical synthetic values, and identical selected lambda. Bootstrap uses its own
  `RandomState` seed; `--seed` seeds the harness draws.
- Outputs: `out/results.json`, `out/results.csv`, `out/run.log`, `out/official_scorer.log`,
  `out/official_reports/`, `out/submissions/` (all under gitignored `out/`). The committed
  `results/` copy holds only the two summary files.
- No network access, no absolute paths baked into the script, data path comes from `--data-dir`
  (defaults to `MTAH_DATA_DIR` or `../data`).

## 9. Keputusan

- **Carry forward** the gated evidence layer (`kin` + `geom` + `set_piece` + `miss_gated`) and
  `dpL_final` (`lambda = 32`, the GT-free selection) as the primary submission candidate. It is the
  only candidate selected by a pre-registered rule that uses no GT, and its full-match possession
  count (280) is closest to the count-band centre.
- **Carry forward** `smooth125_dead_final` as the strong comparator and fallback submission: same
  evidence, no DP, fewer assumptions, and it is not worse on the available window.
- **Drop** `lambda = 4` as a real-data default; keep it only as a synthetic-regime reference.
- **Drop** ungated `long_missing` as a main component; keep it as the negative control that shows
  why gating matters.
- **Do not submit** `dp25_all`-style arms chosen by public-window ranking; that act is forbidden and
  the v1 report's carry-forward verdict is superseded by this one.
- **Repeat** the comparison on full-match GT when it becomes available (post-deadline), and before
  then run only GT-free checks: submission validation, determinism, self-supervised masking probe,
  and cross-check of the D mask against Smart Tagging restarts.
- **Next experiment** (pre-registered, GT-free): calibrate the synthetic confusion generator to the
  measured nearest-player switch rate and presence statistics and re-derive the lambda optimum; if
  it lands near 32 independently of the count rule, the two criteria corroborate each other.
