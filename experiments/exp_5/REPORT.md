# exp_5 — touch-only SetSeq, fresh-seed replication

**Result: H1 passed, narrowly, on both pre-registered clauses. One file is written: `predictions/exp_5_candidate_provider.csv`. No ctrl file: `L_ctrl` non-inferiority failed, so exp_3's `L_ctrl` candidate stays the team's `L_ctrl` option.**

- **Primary.** `exp5_touch_ens3` vs `exp3_ref_ens3`, provider labels, `target_like`:
  - Δ spell-F1 **+0.0102**, 95 % CI **[+0.0009, +0.0199]**, wins 5/7.
  - CI clause: lower bound +0.0009 > 0 → **pass** (margin 0.0009).
  - Seed clause: Δ 0.0102 > `exp5_touch` seed spread **0.0092** → **pass** (margin 0.0010).
- **The replication is weaker than the original.** exp_4's touch ensemble gave +0.015 [+0.007, +0.027], 7/7. The fresh seeds give +0.010; paired exp5 − exp4 is −0.005 [−0.012, +0.002], 1/7 wins. Read: the gain on provider labels holds on these matches, but exp_4's point estimate was inflated by selection (it was the best of 8 ablations). [Medium] — 7 matches, both clauses pass by ~0.001.
- **Provider-specific.** On the other label sets the fresh seeds show no gain: `L_track(1 s)` +0.002, `L_ctrl` −0.008, `L_track(3 s)` −0.002 (all `target_like`, all CIs cross 0).
- **`L_ctrl` gate** (ctrl file only): Δ −0.0083 [−0.0204, +0.0041]; the lower bound is below −0.01 → no ctrl file.
- **Scope.** This is a seed replication on the same 7 IDSSE matches that selected the arm. A pass rules out seed luck; it does not rule out selection on these matches and says nothing about BAR-ATH.
- **BAR-ATH warning (report only, §9).** The written file is valid and rebuilds byte-identically. Its label-free aggregates, though, depart strongly from exp_3's files: dead share 0.488 against a published prior of 0.407, and 2.50 possessions/min against exp_3's provider hedge at 3.75. The model's own mean p(dead) on BAR-ATH is 0.474 against 0.398 for the reference, which suggests the touch features shift on real tactical data. [Medium] Do not swap it in for exp_3's files without checking that shift.

Numbers: IDSSE proxy, 7 matches, nested leave-one-match-out, decoder re-selected per label set, paired match bootstrap (1000 draws, seed 20261108, 95 % percentile). Δ is always `a − exp3_ref_ens3` in mean spell-F1 unless stated. "tl" = `target_like`, "orig" = `original`.

## 1. What was tested

| arm | inputs | loss | seeds | fits |
|---|---|---|---|---|
| `exp5_touch` (→ `exp5_touch_ens3`) | exp_1 tokens + exp_3 touch globals | state + boundary (aux off) | **20261111 / 12 / 13** | 3 × 28 |
| `exp3_ref_ens3` (reference) | exp_1 tokens | state + boundary | exp_2's three seeds | 0 (exp_2 cache, exp_3 results) |
| `exp4_touch_ens3` (replication context) | as `exp5_touch` | as `exp5_touch` | 20261108 / 09 / 10 | 0 (exp_4 cache and results) |
| `exp45_touch_ens6` (report only) | 6-seed mean of the two above | | | 0 |

- Recipe identical to exp_4's `exp4_touch`: SetSeq (DeepSets + BiGRU), `TrainConfig()`, `exp2_aug` rounds, mirror on, provider training targets. Only the seeds are new; they appear nowhere in exp_0–exp_4.
- exp_4 code is used only by import of its pure pieces (`train_tokens`, `view_tokens`, `globals_context`, `exp4_net.train` with `AUX_OFF`, `state_nll`). exp_5 has its own arm table, fingerprints (hashing every fit module, exp_4's included) and folders.
- **Pre-registration.** `PREREG.md` sha256 `dc28ebfcacfa2676c0967d0c0407fa47996b1bea078684a3f6dc73e4d102e814`, frozen 2026-11-10T09:06:10Z before any fit. §0 item 5 records everything seen and clarified before freezing; §1–§8 are as drafted. Every notebook re-verified the hash, the constants and the SHA-256 of every reused exp_3 / exp_4 file.

## 2. Primary decision (H1)

| rule | labels | view | Δ | 95 % CI | wins | criterion | result |
|---|---|---|---|---|---|---|---|
| H1 clause 1 | provider | target_like | +0.0102 | [+0.0009, +0.0199] | 5/7 (0 ties) | paired 95 % CI lower bound > 0 | **pass** |
| H1 clause 2 | provider | target_like | +0.0102 | | | Δ > `exp5_touch` seed spread (0.0092) | **pass** |
| ctrl file gate | ctrl | target_like | −0.0083 | [−0.0204, +0.0041] | 3/7 | paired 95 % CI lower bound > −0.01 | **fail** |

- Both clauses pass by about 0.001. Another bootstrap seed, or one match moving by ~0.007, could flip either clause. The decision follows the frozen rule; its fragility is part of the result.
- **Adoption tree.** H1 passed → 3 full-data fits, `exp_5_candidate_provider.csv` with the decoder selected on the 7 `target_like` out-of-fold predictions against provider labels (T 1.115, λ_team 0.5, λ_dead 4, γ 0), official validator, byte-identical rebuild (§9). No ctrl file.
- **Never adopted:** `exp45_touch_ens6`, `exp4_touch_ens3`, the single seeds.

## 3. S1 — both views, four label sets

Δ and CI against `exp3_ref_ens3` on the same label set and view; temperature and decoder re-selected per label set on the six inner matches. Single seeds are evaluated on provider and `L_ctrl` only. Context rows (`baseline`, `ryu_final`, `rasya_c08`) are exp_3's cached rows.

### provider

| variant | F1 tl | Δ tl | 95 % CI tl | wins tl | count err tl | F1 orig | Δ orig | 95 % CI orig | wins orig | count err orig |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp5_touch_ens3` | 0.674 | +0.010 | [+0.001, +0.020] | 5/7 | -0.20 | 0.676 | +0.010 | [-0.000, +0.020] | 4/7 | -0.20 |
| `exp3_ref_ens3` | 0.664 | ref |  |  | -0.20 | 0.666 | ref |  |  | -0.20 |
| `exp4_touch_ens3` | 0.679 | +0.015 | [+0.007, +0.027] | 7/7 | -0.22 | 0.681 | +0.015 | [+0.006, +0.025] | 6/7 | -0.22 |
| `exp45_touch_ens6` | 0.679 | +0.015 | [+0.006, +0.026] | 7/7 | -0.21 | 0.680 | +0.014 | [+0.005, +0.024] | 6/7 | -0.21 |
| `exp2_aug` | 0.657 | -0.007 | [-0.014, -0.001] | 2/7 | -0.20 | 0.657 | -0.009 | [-0.016, -0.002] | 2/7 | -0.20 |
| `exp5_touch` | 0.668 | +0.004 | [-0.004, +0.012] | 4/7 | -0.21 | 0.671 | +0.005 | [-0.004, +0.012] | 4/7 | -0.20 |
| `exp5_touch_s1` | 0.668 | +0.004 | [-0.005, +0.016] | 3/7 | -0.21 | 0.666 | +0.000 | [-0.013, +0.014] | 3/7 | -0.22 |
| `exp5_touch_s2` | 0.659 | -0.005 | [-0.014, +0.004] | 2/7 | -0.19 | 0.665 | -0.001 | [-0.012, +0.010] | 3/7 | -0.19 |
| `ryu_final` | 0.479 | -0.185 | [-0.221, -0.147] | 0/7 | -0.45 | 0.487 | -0.179 | [-0.217, -0.138] | 0/7 | -0.46 |
| `rasya_c08` | 0.266 | -0.398 | [-0.421, -0.372] | 0/7 | -0.52 | 0.325 | -0.341 | [-0.363, -0.318] | 0/7 | -0.41 |
| `baseline` | 0.204 | -0.460 | [-0.485, -0.436] | 0/7 | +0.23 | 0.186 | -0.480 | [-0.503, -0.456] | 0/7 | +0.18 |

### L_track(1 s)

| variant | F1 tl | Δ tl | 95 % CI tl | wins tl | count err tl | F1 orig | Δ orig | 95 % CI orig | wins orig | count err orig |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp5_touch_ens3` | 0.710 | +0.002 | [-0.008, +0.011] | 4/7 | -0.16 | 0.715 | +0.008 | [-0.003, +0.020] | 4/7 | -0.14 |
| `exp3_ref_ens3` | 0.709 | ref |  |  | -0.14 | 0.707 | ref |  |  | -0.14 |
| `exp4_touch_ens3` | 0.716 | +0.007 | [-0.003, +0.016] | 5/7 | -0.16 | 0.721 | +0.013 | [+0.007, +0.020] | 6/7 | -0.16 |
| `exp45_touch_ens6` | 0.715 | +0.006 | [-0.004, +0.019] | 4/7 | -0.14 | 0.718 | +0.011 | [+0.000, +0.024] | 5/7 | -0.15 |
| `exp2_aug` | 0.698 | -0.011 | [-0.025, +0.002] | 3/7 | -0.16 | 0.702 | -0.005 | [-0.017, +0.006] | 2/7 | -0.15 |
| `ryu_final` | 0.539 | -0.170 | [-0.204, -0.137] | 0/7 | -0.34 | 0.548 | -0.159 | [-0.195, -0.124] | 0/7 | -0.35 |
| `rasya_c08` | 0.297 | -0.411 | [-0.430, -0.391] | 0/7 | -0.43 | 0.356 | -0.351 | [-0.366, -0.338] | 0/7 | -0.30 |
| `baseline` | 0.218 | -0.491 | [-0.508, -0.475] | 0/7 | +0.47 | 0.198 | -0.509 | [-0.525, -0.492] | 0/7 | +0.41 |

### L_ctrl

| variant | F1 tl | Δ tl | 95 % CI tl | wins tl | count err tl | F1 orig | Δ orig | 95 % CI orig | wins orig | count err orig |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp5_touch_ens3` | 0.740 | -0.008 | [-0.020, +0.004] | 3/7 | -0.11 | 0.745 | -0.005 | [-0.016, +0.007] | 3/7 | -0.11 |
| `exp3_ref_ens3` | 0.748 | ref |  |  | -0.13 | 0.751 | ref |  |  | -0.11 |
| `exp4_touch_ens3` | 0.745 | -0.003 | [-0.020, +0.013] | 3/7 | -0.11 | 0.747 | -0.003 | [-0.018, +0.011] | 3/7 | -0.11 |
| `exp45_touch_ens6` | 0.741 | -0.007 | [-0.024, +0.011] | 3/7 | -0.12 | 0.742 | -0.009 | [-0.023, +0.006] | 3/7 | -0.13 |
| `exp2_aug` | 0.735 | -0.014 | [-0.026, +0.002] | 1/7 | -0.10 | 0.730 | -0.021 | [-0.035, -0.007] | 2/7 | -0.11 |
| `exp5_touch` | 0.733 | -0.015 | [-0.033, +0.004] | 1/7 | -0.11 | 0.740 | -0.011 | [-0.029, +0.007] | 3/7 | -0.10 |
| `exp5_touch_s1` | 0.733 | -0.015 | [-0.027, -0.004] | 1/7 | -0.12 | 0.741 | -0.010 | [-0.022, +0.002] | 3/7 | -0.10 |
| `exp5_touch_s2` | 0.729 | -0.019 | [-0.027, -0.011] | 1/7 | -0.11 | 0.732 | -0.018 | [-0.028, -0.007] | 1/7 | -0.11 |
| `ryu_final` | 0.619 | -0.129 | [-0.156, -0.103] | 0/7 | -0.17 | 0.629 | -0.122 | [-0.157, -0.089] | 0/7 | -0.18 |
| `rasya_c08` | 0.304 | -0.444 | [-0.463, -0.421] | 0/7 | -0.29 | 0.365 | -0.386 | [-0.409, -0.361] | 0/7 | -0.11 |
| `baseline` | 0.219 | -0.529 | [-0.543, -0.514] | 0/7 | +0.86 | 0.198 | -0.552 | [-0.569, -0.532] | 0/7 | +0.78 |

### L_track(3 s)

| variant | F1 tl | Δ tl | 95 % CI tl | wins tl | count err tl | F1 orig | Δ orig | 95 % CI orig | wins orig | count err orig |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp5_touch_ens3` | 0.759 | -0.002 | [-0.020, +0.017] | 3/7 | -0.05 | 0.762 | +0.003 | [-0.014, +0.021] | 4/7 | -0.03 |
| `exp3_ref_ens3` | 0.762 | ref |  |  | -0.04 | 0.759 | ref |  |  | -0.02 |
| `exp4_touch_ens3` | 0.763 | +0.002 | [-0.010, +0.013] | 5/7 | -0.05 | 0.765 | +0.006 | [-0.004, +0.014] | 6/7 | -0.04 |
| `exp45_touch_ens6` | 0.769 | +0.008 | [-0.008, +0.026] | 4/7 | -0.06 | 0.774 | +0.015 | [+0.002, +0.030] | 6/7 | -0.06 |
| `exp2_aug` | 0.752 | -0.010 | [-0.024, +0.005] | 2/7 | -0.05 | 0.749 | -0.010 | [-0.021, +0.001] | 2/7 | -0.05 |
| `ryu_final` | 0.656 | -0.106 | [-0.136, -0.069] | 0/7 | +0.01 | 0.666 | -0.093 | [-0.122, -0.057] | 0/7 | -0.01 |
| `rasya_c08` | 0.290 | -0.471 | [-0.485, -0.456] | 0/7 | -0.13 | 0.351 | -0.408 | [-0.423, -0.387] | 0/7 | +0.08 |
| `baseline` | 0.203 | -0.559 | [-0.576, -0.541] | 0/7 | +1.26 | 0.183 | -0.576 | [-0.594, -0.556] | 0/7 | +1.16 |

## 4. Replication: exp_5 vs exp_4 touch, per cell (report only)

| labels | view | Δ exp5 | 95 % CI | wins | Δ exp4 | 95 % CI | wins | exp5 − exp4 | 95 % CI | wins | Δ ens6 | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| provider | original | +0.010 | [-0.000, +0.020] | 4/7 | +0.015 | [+0.006, +0.025] | 6/7 | -0.005 | [-0.010, +0.002] | 2/7 | +0.014 | [+0.005, +0.024] |
| provider | target_like | +0.010 | [+0.001, +0.020] | 5/7 | +0.015 | [+0.007, +0.027] | 7/7 | -0.005 | [-0.012, +0.002] | 1/7 | +0.015 | [+0.006, +0.026] |
| track_1 | original | +0.008 | [-0.003, +0.020] | 4/7 | +0.013 | [+0.007, +0.020] | 6/7 | -0.006 | [-0.011, +0.002] | 1/7 | +0.011 | [+0.000, +0.024] |
| track_1 | target_like | +0.002 | [-0.008, +0.011] | 4/7 | +0.007 | [-0.003, +0.016] | 5/7 | -0.005 | [-0.009, -0.001] | 1/7 | +0.006 | [-0.004, +0.019] |
| ctrl | original | -0.005 | [-0.016, +0.007] | 3/7 | -0.003 | [-0.018, +0.011] | 3/7 | -0.002 | [-0.011, +0.006] | 3/7 | -0.009 | [-0.023, +0.006] |
| ctrl | target_like | -0.008 | [-0.020, +0.004] | 3/7 | -0.003 | [-0.020, +0.013] | 3/7 | -0.005 | [-0.014, +0.005] | 1/7 | -0.007 | [-0.024, +0.011] |
| track_3 | original | +0.003 | [-0.014, +0.021] | 4/7 | +0.006 | [-0.004, +0.014] | 6/7 | -0.003 | [-0.021, +0.012] | 3/7 | +0.015 | [+0.002, +0.030] |
| track_3 | target_like | -0.002 | [-0.020, +0.017] | 3/7 | +0.002 | [-0.010, +0.013] | 5/7 | -0.004 | [-0.020, +0.011] | 3/7 | +0.008 | [-0.008, +0.026] |

- **Direction replicates, size shrinks.** The exp5 Δ has the same sign as exp4's in 7 of 8 cells and is smaller in all 8. The paired exp5 − exp4 delta is negative in all 8 cells (−0.002 to −0.006); only `L_track(1 s)` tl has a CI clear of 0.
- That is the pattern expected when an arm is chosen as the best of eight comparisons: the chosen estimate regresses toward a smaller true effect. A fair single-number estimate of the touch gain on provider labels is about +0.010 to +0.015. [Medium]
- **Pooled `exp45_touch_ens6`** gives +0.015 [+0.006, +0.026], 7/7 on provider tl. It is reported only: half its seeds were seen when this arm was selected, and 6-seed ensembling is a different variant from the one tested.

## 5. Single seeds and seed spread

| family | labels | view | seed 20261111 | seed 20261112 | seed 20261113 | mean | spread (max − min) |
|---|---|---|---|---|---|---|---|
| exp5_touch | provider | original | 0.671 | 0.666 | 0.665 | 0.667 | 0.006 |
| exp5_touch | provider | target_like | 0.668 | 0.668 | 0.659 | 0.665 | **0.009** |
| exp5_touch | ctrl | original | 0.740 | 0.741 | 0.732 | 0.738 | 0.009 |
| exp5_touch | ctrl | target_like | 0.733 | 0.733 | 0.729 | 0.732 | 0.004 |

- exp_4's touch spreads on the same cells were 0.008 / 0.011 / 0.008 / 0.010 (exp_4 `REPORT.md` §5). The seed noise of this recipe is about 0.004–0.011 per 3-seed range.
- Single seeds vs `exp3_ref_ens3` are all inconclusive on provider (−0.005 to +0.005). On `L_ctrl`, s1 tl and s2 (both views) are "worse". A single touch seed is not better than the 3-seed reference; the gain needs the ensemble (exp_3 measured an ensembling gain of +0.014 to +0.018).
- Mean single-seed provider tl: exp5 0.665 vs exp4 0.660, so the fresh seeds are not individually weaker. Their ensemble gains less (+0.009 over the seed mean vs +0.019 for exp4).

## 6. Generalisation gap (report only)

| arm | train NLL | held-out NLL orig | held-out NLL tl | gap orig | gap tl | growth vs exp_2 recipe (orig) | growth (tl) | flag (> 25 %, orig) |
|---|---|---|---|---|---|---|---|---|
| exp2_aug | 0.1165 | 0.1888 | 0.1919 | 0.0722 | 0.0754 | +0.5% | +0.8% |  |
| exp2_aug_s1 | 0.1161 | 0.1884 | 0.1916 | 0.0723 | 0.0754 | +0.5% | +0.9% |  |
| exp2_aug_s2 | 0.1169 | 0.1881 | 0.1903 | 0.0712 | 0.0734 | -1.0% | -1.8% |  |
| exp5_touch | 0.1221 | 0.1756 | 0.1814 | 0.0535 | 0.0593 | -25.6% | -20.6% |  |
| exp5_touch_s1 | 0.1235 | 0.1765 | 0.1828 | 0.0530 | 0.0593 | -26.3% | -20.6% |  |
| exp5_touch_s2 | 0.1255 | 0.1788 | 0.1838 | 0.0533 | 0.0583 | -25.8% | -22.0% |  |
| **exp2_recipe** (family) | 0.1165 | 0.1884 | 0.1913 | 0.0719 | 0.0747 | +0.0% | +0.0% | no |
| **exp5_touch** (family) | 0.1237 | 0.1770 | 0.1827 | 0.0533 | 0.0590 | -25.9% | -21.1% | no |
| exp4_touch (family, exp_4, read in place) | 0.1237 | 0.1764 | 0.1818 | 0.0527 | 0.0581 | -26.6% | | no |

- Gap = held-out NLL − final training NLL (raw p, T = 1, provider targets). Reference training NLL is exp_4's `ref_train_nll.csv`; held-out NLL of the reference comes from exp_2's cached predictions.
- **No flag.** The touch family *shrinks* the gap by ~26 % (orig) / ~21 % (tl), the same as exp_4 (−26.6 %). The rise in training NLL (0.117 → 0.124) and the fall in held-out NLL (0.188 → 0.177 orig) both replicate with fresh seeds. The cause is still untested. [Low]
- The flag uses the `original`-view gap, as exp_4's code did (PREREG §0 item 5); tl growth is shown for completeness.

## 7. Count gates

exp_0 gates against `baseline` on the same view and labels: gain ≥ 0.005, CI lower bound > 0, macro-F1 regression ≤ 0.01, every match's possession count within 30 %. Each cell: pass/fail and the worst |count error| with its match.

| variant | provider tl | provider orig | ctrl tl | ctrl orig |
|---|---|---|---|---|
| `exp5_touch_ens3` | fail: counts (0.33 J03WMX) | fail: counts (0.32 J03WMX) | pass (0.19 J03WR9) | pass (0.19 J03WR9) |
| `exp3_ref_ens3` | fail: counts (0.37 J03WMX) | fail: counts (0.36 J03WMX) | pass (0.22 J03WR9) | pass (0.19 J03WR9) |
| `exp4_touch_ens3` | fail: counts (0.36 J03WMX) | fail: counts (0.35 J03WMX) | pass (0.18 J03WN1) | pass (0.18 J03WR9) |
| `exp45_touch_ens6` | fail: counts (0.32 J03WMX) | fail: counts (0.32 J03WMX) | pass (0.18 J03WMX) | pass (0.20 J03WN1) |
| `exp2_aug` | fail: counts (0.35 J03WMX) | fail: counts (0.35 J03WMX) | pass (0.22 J03WR9) | pass (0.22 J03WR9) |
| `exp5_touch` | fail: counts (0.31 J03WMX) | fail: counts (0.31 J03WMX) | pass (0.18 J03WN1) | pass (0.19 J03WR9) |
| `exp5_touch_s1` | fail: counts (0.33 J03WMX) | fail: counts (0.34 J03WMX) | pass (0.19 J03WN1) | pass (0.17 J03WN1) |
| `exp5_touch_s2` | fail: counts (0.34 J03WMX) | pass (0.30 J03WMX) | pass (0.20 J03WN1) | pass (0.17 J03WN1) |
| `ryu_final` | fail: counts (0.52 J03WMX) | fail: counts (0.53 J03WMX) | pass (0.22 J03WOH) | pass (0.22 J03WOH) |
| `rasya_c08` | fail: macro, counts (0.62 J03WMX) | fail: counts (0.49 J03WMX) | fail: macro, counts (0.35 J03WMX) | pass (0.15 J03WOH) |

- As in exp_4, every model arm fails only the count gate on provider labels, always on J03WMX (under-count 31–37 %). `exp5_touch_ens3` is the closest 3-seed ensemble so far (0.33 / 0.32) but still fails. Out of scope for exp_5 (PROMPT).
- The one provider pass (`exp5_touch_s2` orig, 0.30) sits on the limit and is a single seed; it means nothing.
- `L_ctrl`: every exp_5 arm passes all gates.

## 8. Argmax diagnostic and short possessions

| ensemble | labels | view | F1 decoded | F1 argmax | argmax − decoded | count err decoded | count err argmax |
|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | provider | target_like | 0.664 | 0.643 | -0.022 | -0.20 | -0.07 |
| `exp5_touch_ens3` | provider | target_like | 0.674 | 0.651 | -0.024 | -0.20 | -0.08 |
| `exp3_ref_ens3` | provider | original | 0.666 | 0.648 | -0.019 | -0.20 | -0.07 |
| `exp5_touch_ens3` | provider | original | 0.676 | 0.651 | -0.025 | -0.20 | -0.09 |
| `exp3_ref_ens3` | ctrl | target_like | 0.748 | 0.631 | -0.117 | -0.13 | +0.40 |
| `exp5_touch_ens3` | ctrl | target_like | 0.740 | 0.641 | -0.099 | -0.11 | +0.38 |
| `exp3_ref_ens3` | ctrl | original | 0.751 | 0.635 | -0.115 | -0.11 | +0.40 |
| `exp5_touch_ens3` | ctrl | original | 0.745 | 0.645 | -0.100 | -0.11 | +0.38 |

- The touch ensemble's raw argmax beats the reference's on all four cells (+0.003 to +0.010), so part of the gain is in the per-step probabilities, not only in decoding.
- The decoder still matters more than the model change: argmax loses ~0.02 F1 on provider and 0.10–0.12 on ctrl, and over-counts by ~+38 % on ctrl.

| variant | view | recall, all possessions (decoded) | recall < 2 s (decoded) | recall < 2 s (argmax) |
|---|---|---|---|---|
| `exp2_aug` | original | 0.607 | 0.289 | 0.326 |
| `exp3_ref_ens3` | original | 0.610 | 0.302 | 0.348 |
| `exp4_touch_ens3` | original | 0.618 | 0.298 | 0.331 |
| `exp5_touch_ens3` | original | 0.616 | 0.318 | 0.348 |
| `exp45_touch_ens6` | original | 0.617 | 0.310 | 0.339 |
| `exp2_aug` | target_like | 0.607 | 0.287 | 0.322 |
| `exp3_ref_ens3` | target_like | 0.609 | 0.301 | 0.342 |
| `exp4_touch_ens3` | target_like | 0.615 | 0.302 | 0.327 |
| `exp5_touch_ens3` | target_like | 0.614 | 0.314 | 0.345 |
| `exp45_touch_ens6` | target_like | 0.615 | 0.315 | 0.340 |

- Provider labels, IoU 0.5, short = true possession < 2 s. Touch ensembles recover slightly more possessions overall (0.614–0.618 vs 0.609–0.610). Short-possession recall is +0.013 to +0.016 for exp5 but ~0 for exp4, so it is within seed noise. About 70 % of short possessions are still missed by every arm.

## 9. BAR-ATH (adopted file; aggregates only)

Notebook 04 (`results/target_candidates.json`). The input is **tactical**: exp_3's pinned fusion rule rejected fusion, as in exp_3. There are 3 full-data `exp5_touch` fits (seeds 20261111 / 12 / 13 on all 7 IDSSE matches), and the decoder is the final provider config (T 1.115, λ_team 0.5, λ_dead 4, γ 0).

| file | sha256 | valid | dead share | − published prior (0.407) | possessions / min | mean possession (s) | possessions | model mean p(D) |
|---|---|---|---|---|---|---|---|---|
| `exp_5_candidate_provider.csv` | `1b9795cb…42da62` | yes | 0.488 | **+0.081** | 2.50 | 12.3 | 265 | 0.474 |
| exp_3 `target_hedge_provider` (`exp3_ref_ens3`, provider decoder) | `a6ebb94b…cd96f3` | yes | 0.408 | +0.001 | 3.75 | 9.5 | 397 | 0.398 |
| exp_3 `target_candidate` (`exp3_ref_ens3`, `L_ctrl` decoder) | `17eefbcc…737c08` | yes | 0.410 | +0.003 | 2.45 | 14.5 | 259 | 0.398 |

- **Rebuild.** 3 independent full-data fits (no cached weights) rebuilt the file **byte-identically** (same sha256). The official validator passed (`X1.write_csv`).
- **Plausibility warning (report only; PREREG §5 forbids acting on it). [Medium]**
  - On BAR-ATH, the touch ensemble's own mean p(dead) is 0.474, against 0.398 for the reference ensemble on the same tactical input.
  - Decoded with a near-identical provider decoder, that is +8 points of dead ball over the published prior and the reference, and 33 % fewer possessions than exp_3's provider hedge (2.50 vs 3.75 / min).
  - On IDSSE the two ensembles have the same provider count error (−0.20 both), so the shift appears only on BAR-ATH.
  - Likely cause [Low, untested]: the 4 touch features are computed from the tactical ball (visible on 59 % of graded frames, with its own noise), whose distribution differs from the simulated `target_like` degradation the model was trained and validated on. A touch-feature distribution check on BAR-ATH vs `target_like` would test this; it needs a new pre-registration.
- **Consequence for the team.** The rule required writing this file, and its proxy gain is +0.010. Its label-free BAR-ATH behaviour, however, departs from both exp_3 files and from the published dead-share prior by much more than anything seen on the proxy. It should not replace exp_3's files without that check. The choice between files is the team's; exp_5 makes no BAR-ATH selection.

## 10. Limitations

- **Same 7 matches.** exp_5 re-tests the arm on the matches that selected it. A pass rules out seed luck, not selection on these matches or overfitting of the whole exp_1–exp_5 series to them. Nothing here tests transfer to BAR-ATH; the public check is spent.
- **Marginal pass.** Both clauses pass by ~0.001. With 7 bootstrap units, the decision is sensitive to a single match. The honest reading is "small positive effect on provider labels, about +0.01", not "touch is clearly better".
- **The gain is label-set specific.** No gain on `L_ctrl` (−0.008) or the track labels. exp_3's primary BAR-ATH candidate uses `L_ctrl`; exp_5 does not improve it, and its ctrl file was not written.
- **Seed noise is estimated from 3 seeds.** The spread used in clause 2 (0.0092) is itself a 3-sample range; exp_4's equivalent was 0.011. With exp_4's value, clause 2 would have failed (0.0102 < 0.011). That was not the pre-registered rule, but it shows how close the call is.
- **Parallel fit processes (§11).** Seeds 2–3 ran in 2–3 concurrent processes. Each fit seeds itself and uses deterministic algorithms, so results should not depend on scheduling, but only the single-process reproduction check tested bit-exactness. [Medium]
- **Touch features may not transfer to BAR-ATH (§9).** On the real tactical input the touch ensemble predicts about 8 points more dead ball than the reference. The proxy cannot see this, because `target_like` is a simulation of BAR-ATH's degradation, not BAR-ATH.
- **Training-NLL rise with touch features is still unexplained (§6).**
- **The J03WMX provider count problem** is unchanged and out of scope.

## 11. Reproduction and hygiene

- **Notebooks.** `00_protocol` → `04_target` executed via `run_notebook.py`; each verifies the PREREG sha256, the pre-registered constants and the SHA-256 of all reused exp_3 / exp_4 files first.
- **Reproduction check** (`results/repro_check.json`): exp_5's code path with exp_4's seed 20261108, outer fold J03WMX, is **bit-exact** to exp_4's cached prediction on both views (max |Δp| 0, max |Δb| 0; 52 s).
- **Reference rechecks** (`results/reference_recheck.json`): `exp3_ref_ens3` (0.6641) and `exp4_touch_ens3` (0.6793), provider / tl, recomputed from cached predictions, equal the stored rows and final configs exactly.
- **Tests.** 16 synthetic tests (`tests/test_exp5.py`) pass: arm table (fresh seeds, touch on, `AUX_OFF`, `TrainConfig()`), a tiny fit + evaluation with patched paths writes nothing under exp_0–exp_4 or `data/`, `fit=False` refusal, both decision clauses, the ctrl-file gate, fingerprint changes for every hashed fit module, freeze refusal once fits exist.
- **Fits.** 84 proxy + 1 reproduction check + 3 full-data + 3 rebuild = 91 of a cap of 100 (§9). Seed 20261111 took 1446 s in one process (~52 s/fit).
- **Process incident.** The app stopped the first notebook-01 background task during seed 20261112 (1 fit in progress, nothing saved). While restarting, two launch attempts that looked dead in `ps` survived, so seeds 2–3 ran in three, then two, concurrent processes sharing the per-job locks. Each job was fitted and saved exactly once (28 `.npz` + 28 `.pt` per arm, 84 `FIT` lines). Two interrupted fit attempts (the stopped task, and one extra process stopped to free RAM) saved nothing and are not in the count; with them the total stays ≤ 93. A first notebook-04 run was stopped by a session restart before reading any BAR-ATH data or fitting; orphaned idle kernels were stopped and 04 was re-run.
- **Write guard** (`results/write_guard_03.json`, `write_guard_04.json`): file count + size/mtime digest of exp_0–exp_4 and `data/` unchanged since freezing (no file content read).
- **Data hygiene.** No executed notebook was opened. `evaluation/gt/` was not opened. BAR-ATH was read only inside notebook 04's kernel, and only aggregates were printed or stored. Nothing under `data/` or exp_0–exp_4 was written. exp_0–exp_4 were imported, never edited.

## Files

- `results/decision.json`: H1 with both clauses, the ctrl gate, adoption and every report-only delta.
- `results/proxy_report.json`: full table, gates, replication, gap table, short recall, argmax, reproduction check and rechecks.
- `results/comparison_table.csv`, `per_match_results.csv`, `replication_table.csv`, `fig_exp5.png`.
- `results/gap_table.csv`, `train_nll.csv`, `heldout_nll.csv`, `argmax_rows.csv`, `short_recall.csv`.
- `results/target_candidates.json`: BAR-ATH aggregates, validator, rebuild.
- `predictions/exp_5_candidate_provider.csv` (gitignored).
