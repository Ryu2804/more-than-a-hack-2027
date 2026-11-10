# exp_4 — touch features + auxiliary hidden-ball head

**Result: H1 failed. No exp_4 BAR-ATH files are written; exp_3's two files stay the team's options.**

- **Primary.** `exp4_full_ens3` vs `exp3_ref_ens3`, provider labels, `target_like` view:
  - Δ spell-F1 is +0.007, 95 % CI [−0.005, +0.019]. The lower bound is not > 0.
  - On `L_ctrl` the non-inferiority bound also fails: Δ −0.006, CI [−0.016, +0.003], lower bound below −0.01.
  - The provider Δ (+0.007) is smaller than the `exp4_full` seed spread (0.0167), so it is **noise**.
- **Report only, not adopted.** Touch features alone (`exp4_touch_ens3`) beat the reference on provider labels:
  - `target_like`: +0.015 [+0.007, +0.027], 7/7 matches;
  - `original`: +0.015 [+0.006, +0.025];
  - no gain on `L_ctrl`: −0.003 [−0.020, +0.013].

  This is one of eight report-only ablation comparisons. The fixed-arm rule (PREREG §5) forbids adopting it. It is a candidate for its own pre-registration (see §9).
- **The auxiliary head added nothing measurable.**
  - `exp4_aux_ens3` vs reference: provider +0.003 [−0.003, +0.010]; ctrl −0.003 [−0.011, +0.004].
  - Stacked on touch (`exp4_full` vs `exp4_touch`): provider −0.008 [−0.016, +0.001].
  - Recall of possessions < 2 s did not move (§6).
- **Overfitting controls held.**
  - PREREG hash was verified in all five notebooks.
  - No family's train→held-out NLL gap grew; the touch families' gap shrank ~27 %.
  - The bit-exact reproduction check passed.
  - 253 of 280 fits used.

Numbers: IDSSE proxy, 7 matches, nested leave-one-match-out, paired match bootstrap (1000 draws, seed 20261108, 95 % percentile). Δ is always `a − exp3_ref_ens3` in mean spell-F1 unless stated. "tl" = `target_like`, "orig" = `original`.

## 1. What was tested

| arm | inputs | loss | seeds | fits |
|---|---|---|---|---|
| `exp4_full` | exp_1 tokens + exp_3 touch globals | state + boundary + aux (ball xy smooth-L1 0.25, nearest team CE 0.25) | 20261108 / 09 / 10 | 3 × 28 |
| `exp4_touch` | exp_1 tokens + exp_3 touch globals | state + boundary | same | 3 × 28 |
| `exp4_aux` | exp_1 tokens | state + boundary + aux | same | 3 × 28 |
| `exp3_ref_ens3` (reference) | exp_1 tokens | state + boundary | exp_2's three seeds | reused, cached |

- **Shared setup.** Every arm uses SetSeq (DeepSets + BiGRU), `TrainConfig()`, the `target_aug` recipe and provider training labels.
- **Aux head.**
  - A linear layer on the BiGRU state, used in training only.
  - Targets come from the **clean** IDSSE match: ball xy, and the team of the player nearest the ball.
  - The flip sign comes from the observed corrupted round.
  - Held-out arrays never carry targets (test `test_aux_targets_come_from_training_matches_only`).
  - At inference the head is unused, so `exp4_aux` and the reference have identical inference inputs.
- **Pre-registration.**
  - `PREREG.md` sha256 `254e2e216cb0515430fba018d849d02a2dfce02f5a24e948e9cbb1d8d1dda975`, frozen 2026-11-09T17:09:15Z, before any fit.
  - It was re-frozen once before any fit to fix an aux sign bug (PREREG §0.5). The old protocol is kept in `results/protocol_superseded_1.json`.

## 2. Primary decision (H1)

| rule | labels | view | Δ | 95 % CI | wins | criterion | result |
|---|---|---|---|---|---|---|---|
| H1_provider | provider | target_like | +0.0074 | [-0.005, +0.019] | 4/7 (0 ties) | paired 95% CI lower bound > 0.0 | **fail** |
| H1_ctrl_non_inferiority | ctrl | target_like | -0.0062 | [-0.016, +0.003] | 3/7 (1 ties) | paired 95% CI lower bound > -0.01 | **fail** |

- **Seed honesty.**
  - The provider Δ (+0.0074) is smaller than the `exp4_full` single-seed spread on the same cell (0.0167). REPORT calls it noise.
  - That spread is larger than the ~0.010 exp_3 estimated.
- **Adoption tree.** H1 failed, so `04_target.ipynb` wrote `results/target_none.json`:
  - `files_written: []`;
  - `bar_ath_read: false`.

  No full-data fits were run.

## 3. S1 — every arm, both views, four label sets

- Δ and CI are against `exp3_ref_ens3` on the same label set and view.
- Temperature and decoder are re-selected per label set on the six inner matches.
- Single seeds were evaluated on the two decision label sets only. `L_track(1 s)` and `L_track(3 s)` cover the ensembles.
- Context rows come from exp_3's cached results: `baseline` (exp_0 heuristic), `ryu_final`, `rasya_c08`.

### provider (4.58 poss/min)

| variant | F1 tl | Δ tl | 95 % CI tl | count err tl | F1 orig | Δ orig | 95 % CI orig | count err orig |
|---|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | 0.664 | ref |  | -0.20 | 0.666 | ref |  | -0.20 |
| `exp4_full_ens3` | 0.672 | +0.007 | [-0.005, +0.019] | -0.21 | 0.674 | +0.007 | [-0.007, +0.026] | -0.22 |
| `exp4_touch_ens3` | 0.679 | +0.015 | [+0.007, +0.027] | -0.22 | 0.681 | +0.015 | [+0.006, +0.025] | -0.22 |
| `exp4_aux_ens3` | 0.667 | +0.003 | [-0.003, +0.010] | -0.21 | 0.667 | +0.001 | [-0.005, +0.007] | -0.22 |
| `exp2_aug` | 0.657 | -0.007 | [-0.014, -0.001] | -0.20 | 0.657 | -0.009 | [-0.016, -0.002] | -0.20 |
| `exp4_full` | 0.658 | -0.006 | [-0.015, +0.003] | -0.21 | 0.662 | -0.005 | [-0.015, +0.006] | -0.20 |
| `exp4_full_s1` | 0.650 | -0.014 | [-0.022, -0.006] | -0.22 | 0.652 | -0.014 | [-0.024, -0.004] | -0.22 |
| `exp4_full_s2` | 0.667 | +0.003 | [-0.009, +0.013] | -0.19 | 0.665 | -0.001 | [-0.014, +0.011] | -0.20 |
| `exp4_touch` | 0.659 | -0.005 | [-0.014, +0.006] | -0.21 | 0.665 | -0.002 | [-0.013, +0.012] | -0.20 |
| `exp4_touch_s1` | 0.655 | -0.009 | [-0.015, -0.004] | -0.21 | 0.659 | -0.007 | [-0.017, +0.002] | -0.21 |
| `exp4_touch_s2` | 0.666 | +0.001 | [-0.012, +0.017] | -0.23 | 0.667 | +0.001 | [-0.010, +0.015] | -0.21 |
| `exp4_aux` | 0.654 | -0.010 | [-0.016, -0.005] | -0.21 | 0.659 | -0.007 | [-0.015, +0.000] | -0.22 |
| `exp4_aux_s1` | 0.646 | -0.018 | [-0.026, -0.009] | -0.22 | 0.653 | -0.013 | [-0.020, -0.006] | -0.21 |
| `exp4_aux_s2` | 0.649 | -0.015 | [-0.022, -0.008] | -0.23 | 0.650 | -0.016 | [-0.025, -0.006] | -0.22 |
| `baseline` | 0.204 | -0.460 | [-0.485, -0.436] | +0.23 | 0.186 | -0.480 | [-0.503, -0.456] | +0.18 |
| `ryu_final` | 0.479 | -0.185 | [-0.221, -0.147] | -0.45 | 0.487 | -0.179 | [-0.217, -0.138] | -0.46 |
| `rasya_c08` | 0.266 | -0.398 | [-0.421, -0.372] | -0.52 | 0.325 | -0.341 | [-0.363, -0.318] | -0.41 |

### L_track(1 s)

| variant | F1 tl | Δ tl | 95 % CI tl | count err tl | F1 orig | Δ orig | 95 % CI orig | count err orig |
|---|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | 0.709 | ref |  | -0.14 | 0.707 | ref |  | -0.14 |
| `exp4_full_ens3` | 0.711 | +0.002 | [-0.009, +0.014] | -0.14 | 0.711 | +0.004 | [-0.010, +0.019] | -0.13 |
| `exp4_touch_ens3` | 0.716 | +0.007 | [-0.003, +0.016] | -0.16 | 0.721 | +0.013 | [+0.007, +0.020] | -0.16 |
| `exp4_aux_ens3` | 0.702 | -0.007 | [-0.017, +0.003] | -0.14 | 0.701 | -0.006 | [-0.018, +0.007] | -0.13 |
| `exp2_aug` | 0.698 | -0.011 | [-0.025, +0.002] | -0.16 | 0.702 | -0.005 | [-0.017, +0.006] | -0.15 |
| `baseline` | 0.218 | -0.491 | [-0.508, -0.475] | +0.47 | 0.198 | -0.509 | [-0.525, -0.492] | +0.41 |
| `ryu_final` | 0.539 | -0.170 | [-0.204, -0.137] | -0.34 | 0.548 | -0.159 | [-0.195, -0.124] | -0.35 |
| `rasya_c08` | 0.297 | -0.411 | [-0.430, -0.391] | -0.43 | 0.356 | -0.351 | [-0.366, -0.338] | -0.30 |

### L_ctrl (3.02 poss/min)

| variant | F1 tl | Δ tl | 95 % CI tl | count err tl | F1 orig | Δ orig | 95 % CI orig | count err orig |
|---|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | 0.748 | ref |  | -0.13 | 0.751 | ref |  | -0.11 |
| `exp4_full_ens3` | 0.742 | -0.006 | [-0.016, +0.003] | -0.13 | 0.747 | -0.004 | [-0.016, +0.007] | -0.13 |
| `exp4_touch_ens3` | 0.745 | -0.003 | [-0.020, +0.013] | -0.11 | 0.747 | -0.003 | [-0.018, +0.011] | -0.11 |
| `exp4_aux_ens3` | 0.745 | -0.003 | [-0.011, +0.004] | -0.12 | 0.744 | -0.006 | [-0.011, -0.001] | -0.14 |
| `exp2_aug` | 0.735 | -0.014 | [-0.026, +0.002] | -0.10 | 0.730 | -0.021 | [-0.035, -0.007] | -0.11 |
| `exp4_full` | 0.739 | -0.010 | [-0.018, -0.001] | -0.12 | 0.739 | -0.012 | [-0.023, -0.001] | -0.11 |
| `exp4_full_s1` | 0.728 | -0.020 | [-0.032, -0.010] | -0.09 | 0.732 | -0.019 | [-0.035, -0.003] | -0.10 |
| `exp4_full_s2` | 0.745 | -0.003 | [-0.022, +0.013] | -0.12 | 0.746 | -0.005 | [-0.026, +0.014] | -0.13 |
| `exp4_touch` | 0.738 | -0.010 | [-0.023, +0.001] | -0.10 | 0.742 | -0.008 | [-0.021, +0.004] | -0.10 |
| `exp4_touch_s1` | 0.730 | -0.019 | [-0.030, -0.006] | -0.13 | 0.734 | -0.016 | [-0.027, -0.007] | -0.12 |
| `exp4_touch_s2` | 0.740 | -0.009 | [-0.031, +0.011] | -0.13 | 0.740 | -0.011 | [-0.031, +0.006] | -0.11 |
| `exp4_aux` | 0.737 | -0.011 | [-0.018, -0.003] | -0.11 | 0.746 | -0.004 | [-0.014, +0.007] | -0.12 |
| `exp4_aux_s1` | 0.734 | -0.014 | [-0.025, -0.005] | -0.13 | 0.733 | -0.017 | [-0.028, -0.006] | -0.12 |
| `exp4_aux_s2` | 0.720 | -0.029 | [-0.037, -0.020] | -0.14 | 0.727 | -0.023 | [-0.034, -0.013] | -0.14 |
| `baseline` | 0.219 | -0.529 | [-0.543, -0.514] | +0.86 | 0.198 | -0.552 | [-0.569, -0.532] | +0.78 |
| `ryu_final` | 0.619 | -0.129 | [-0.156, -0.103] | -0.17 | 0.629 | -0.122 | [-0.157, -0.089] | -0.18 |
| `rasya_c08` | 0.304 | -0.444 | [-0.463, -0.421] | -0.29 | 0.365 | -0.386 | [-0.409, -0.361] | -0.11 |

### L_track(3 s)

| variant | F1 tl | Δ tl | 95 % CI tl | count err tl | F1 orig | Δ orig | 95 % CI orig | count err orig |
|---|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | 0.762 | ref |  | -0.04 | 0.759 | ref |  | -0.02 |
| `exp4_full_ens3` | 0.756 | -0.006 | [-0.019, +0.012] | -0.06 | 0.754 | -0.005 | [-0.018, +0.012] | -0.03 |
| `exp4_touch_ens3` | 0.763 | +0.002 | [-0.010, +0.013] | -0.05 | 0.765 | +0.006 | [-0.004, +0.014] | -0.04 |
| `exp4_aux_ens3` | 0.752 | -0.010 | [-0.016, -0.003] | -0.05 | 0.753 | -0.006 | [-0.015, +0.003] | -0.04 |
| `exp2_aug` | 0.752 | -0.010 | [-0.024, +0.005] | -0.05 | 0.749 | -0.010 | [-0.021, +0.001] | -0.05 |
| `baseline` | 0.203 | -0.559 | [-0.576, -0.541] | +1.26 | 0.183 | -0.576 | [-0.594, -0.556] | +1.16 |
| `ryu_final` | 0.656 | -0.106 | [-0.136, -0.069] | +0.01 | 0.666 | -0.093 | [-0.122, -0.057] | -0.01 |
| `rasya_c08` | 0.290 | -0.471 | [-0.485, -0.456] | -0.13 | 0.351 | -0.408 | [-0.423, -0.387] | +0.08 |

### `exp4_full_ens3` vs reference across S1

| a (vs `exp3_ref_ens3`) | labels | view | Δ | 95 % CI | wins | bootstrap call |
|---|---|---|---|---|---|---|
| `exp4_full_ens3` | provider | original | +0.007 | [-0.007, +0.026] | 3/7 | inconclusive |
| `exp4_full_ens3` | provider | target_like | +0.007 | [-0.005, +0.019] | 4/7 | inconclusive |
| `exp4_full_ens3` | track_1 | original | +0.004 | [-0.010, +0.019] | 3/7 | inconclusive |
| `exp4_full_ens3` | track_1 | target_like | +0.002 | [-0.009, +0.014] | 4/7 | inconclusive |
| `exp4_full_ens3` | ctrl | original | -0.004 | [-0.016, +0.007] | 4/7 | inconclusive |
| `exp4_full_ens3` | ctrl | target_like | -0.006 | [-0.016, +0.003] | 3/7 | inconclusive |
| `exp4_full_ens3` | track_3 | original | -0.005 | [-0.018, +0.012] | 3/7 | inconclusive |
| `exp4_full_ens3` | track_3 | target_like | -0.006 | [-0.019, +0.012] | 2/7 | inconclusive |

`exp4_full_ens3` is inconclusive on all eight cells: positive on the provider and 1 s labels, slightly negative on the ctrl and 3 s labels.

## 4. Paired deltas: ablations and single seeds

| group | a | b | labels | view | Δ | 95 % CI | wins | bootstrap call |
|---|---|---|---|---|---|---|---|---|
| ablation vs ref | `exp4_touch_ens3` | `exp3_ref_ens3` | provider | original | +0.015 | [+0.006, +0.025] | 6/7 | better |
| ablation vs ref | `exp4_touch_ens3` | `exp3_ref_ens3` | provider | target_like | +0.015 | [+0.007, +0.027] | 7/7 | better |
| ablation vs ref | `exp4_touch_ens3` | `exp3_ref_ens3` | ctrl | original | -0.003 | [-0.018, +0.011] | 3/7 | inconclusive |
| ablation vs ref | `exp4_touch_ens3` | `exp3_ref_ens3` | ctrl | target_like | -0.003 | [-0.020, +0.013] | 3/7 | inconclusive |
| ablation vs ref | `exp4_aux_ens3` | `exp3_ref_ens3` | provider | original | +0.001 | [-0.005, +0.007] | 4/7 | inconclusive |
| ablation vs ref | `exp4_aux_ens3` | `exp3_ref_ens3` | provider | target_like | +0.003 | [-0.003, +0.010] | 4/7 | inconclusive |
| ablation vs ref | `exp4_aux_ens3` | `exp3_ref_ens3` | ctrl | original | -0.006 | [-0.011, -0.001] | 1/7 | worse |
| ablation vs ref | `exp4_aux_ens3` | `exp3_ref_ens3` | ctrl | target_like | -0.003 | [-0.011, +0.004] | 2/7 | inconclusive |
| full vs ablation | `exp4_full_ens3` | `exp4_touch_ens3` | provider | target_like | -0.008 | [-0.016, +0.001] | 2/7 | inconclusive |
| full vs ablation | `exp4_full_ens3` | `exp4_touch_ens3` | ctrl | target_like | -0.003 | [-0.011, +0.005] | 4/7 | inconclusive |
| full vs ablation | `exp4_full_ens3` | `exp4_aux_ens3` | provider | target_like | +0.004 | [-0.011, +0.020] | 4/7 | inconclusive |
| full vs ablation | `exp4_full_ens3` | `exp4_aux_ens3` | ctrl | target_like | -0.003 | [-0.011, +0.007] | 3/7 | inconclusive |
| single seed vs ref | `exp4_full` | `exp3_ref_ens3` | provider | target_like | -0.006 | [-0.015, +0.003] | 3/7 | inconclusive |
| single seed vs ref | `exp4_full` | `exp3_ref_ens3` | ctrl | target_like | -0.010 | [-0.018, -0.001] | 2/7 | worse |
| single seed vs ref | `exp4_full_s1` | `exp3_ref_ens3` | provider | target_like | -0.014 | [-0.022, -0.006] | 1/7 | worse |
| single seed vs ref | `exp4_full_s1` | `exp3_ref_ens3` | ctrl | target_like | -0.020 | [-0.032, -0.010] | 0/7 | worse |
| single seed vs ref | `exp4_full_s2` | `exp3_ref_ens3` | provider | target_like | +0.003 | [-0.009, +0.013] | 4/7 | inconclusive |
| single seed vs ref | `exp4_full_s2` | `exp3_ref_ens3` | ctrl | target_like | -0.003 | [-0.022, +0.013] | 4/7 | inconclusive |

- **Touch features are the useful part on provider labels.**
  - Touch alone gives +0.015, with a CI clear of 0 on both views.
  - Touch plus aux gives +0.007, with a CI crossing 0.
- **The aux head never helps significantly.**
  - It is "worse" on ctrl/original (−0.006 [−0.011, −0.001]).
  - On top of touch it moves provider F1 by −0.008 (inconclusive).
- **Single seeds against a 3-seed ensemble are lower, as expected.** exp_3 measured an ensembling gain of +0.014 to +0.018.
- **Multiplicity.**
  - Eight ablation-vs-reference comparisons are run here, so one "better" call by chance is plausible.
  - The touch result is consistent across both views and 7/7 matches (tl), but on the same 7 matches.
  - Strength of the touch-only claim: [Medium].

## 5. Single seeds and seed spread

| family | labels | view | seed 0 | seed 1 | seed 2 | mean | spread (max−min) |
|---|---|---|---|---|---|---|---|
| exp4_full | provider | original | 0.662 | 0.652 | 0.665 | 0.660 | 0.013 |
| exp4_touch | provider | original | 0.665 | 0.659 | 0.667 | 0.664 | 0.008 |
| exp4_aux | provider | original | 0.659 | 0.653 | 0.650 | 0.654 | 0.009 |
| exp4_full | provider | target_like | 0.658 | 0.650 | 0.667 | 0.659 | 0.017 |
| exp4_touch | provider | target_like | 0.659 | 0.655 | 0.666 | 0.660 | 0.011 |
| exp4_aux | provider | target_like | 0.654 | 0.646 | 0.649 | 0.650 | 0.008 |
| exp4_full | ctrl | original | 0.739 | 0.732 | 0.746 | 0.739 | 0.014 |
| exp4_touch | ctrl | original | 0.742 | 0.734 | 0.740 | 0.739 | 0.008 |
| exp4_aux | ctrl | original | 0.746 | 0.733 | 0.727 | 0.736 | 0.019 |
| exp4_full | ctrl | target_like | 0.739 | 0.728 | 0.745 | 0.737 | 0.017 |
| exp4_touch | ctrl | target_like | 0.738 | 0.730 | 0.740 | 0.736 | 0.010 |
| exp4_aux | ctrl | target_like | 0.737 | 0.734 | 0.720 | 0.730 | 0.017 |

- The spread is the max − min of per-seed mean spell-F1 over the 7 matches.
- It is 0.008–0.019, mostly above the ~0.010 exp_3 assumed.
- Any Δ below ~0.017 on the primary cell should be read as within seed noise.
- The touch-only Δ of +0.015 is above the touch family's own spread (0.011) but below the largest observed spread (0.019).

## 6. Argmax diagnostic and short possessions

| ensemble | labels | view | F1 decoded | F1 argmax | argmax − decoded | count err decoded | count err argmax |
|---|---|---|---|---|---|---|---|
| `exp3_ref_ens3` | provider | target_like | 0.664 | 0.643 | -0.022 | -0.20 | -0.07 |
| `exp4_full_ens3` | provider | target_like | 0.672 | 0.644 | -0.028 | -0.21 | -0.08 |
| `exp3_ref_ens3` | provider | original | 0.666 | 0.648 | -0.019 | -0.20 | -0.07 |
| `exp4_full_ens3` | provider | original | 0.674 | 0.643 | -0.031 | -0.22 | -0.07 |
| `exp3_ref_ens3` | ctrl | target_like | 0.748 | 0.631 | -0.117 | -0.13 | +0.40 |
| `exp4_full_ens3` | ctrl | target_like | 0.742 | 0.638 | -0.104 | -0.13 | +0.39 |
| `exp3_ref_ens3` | ctrl | original | 0.751 | 0.635 | -0.115 | -0.11 | +0.40 |
| `exp4_full_ens3` | ctrl | original | 0.747 | 0.638 | -0.108 | -0.13 | +0.39 |

- The decoder matters more than the model change.
- Argmax loses 0.019–0.031 F1 on provider labels and 0.104–0.117 on ctrl labels.
- On ctrl, argmax over-counts by +39 % against −13 % decoded.
- exp_4 changes neither pattern.

| variant | view | recall, all possessions (decoded) | recall < 2 s (decoded) | recall < 2 s (argmax) |
|---|---|---|---|---|
| `exp2_aug` | original | 0.607 | 0.289 | 0.326 |
| `exp3_ref_ens3` | original | 0.610 | 0.302 | 0.348 |
| `exp4_aux_ens3` | original | 0.607 | 0.299 | 0.329 |
| `exp4_full_ens3` | original | 0.610 | 0.301 | 0.327 |
| `exp4_touch_ens3` | original | 0.618 | 0.298 | 0.331 |
| `exp2_aug` | target_like | 0.607 | 0.287 | 0.322 |
| `exp3_ref_ens3` | target_like | 0.609 | 0.301 | 0.342 |
| `exp4_aux_ens3` | target_like | 0.609 | 0.302 | 0.326 |
| `exp4_full_ens3` | target_like | 0.608 | 0.298 | 0.323 |
| `exp4_touch_ens3` | target_like | 0.615 | 0.302 | 0.327 |

- Recall is matched spells over true spells (exp_3 `_matched`); short means a true possession < 2 s.
- About 30 % of short provider possessions are recovered by every arm.
- The aux head, whose motivation was the hidden-ball / short-touch case, does not raise it: `exp4_full_ens3` scores 0.298 against the reference's 0.301 on tl.

## 7. Generalisation gap

| arm | train NLL | held-out NLL orig | held-out NLL tl | gap orig | gap tl | gap change vs exp_2 recipe (tl) | >25 % growth |
|---|---|---|---|---|---|---|---|
| exp2_aug | 0.1165 | 0.1888 | 0.1919 | 0.0722 | 0.0754 | +0.5 % | – |
| exp2_aug_s1 | 0.1161 | 0.1884 | 0.1916 | 0.0723 | 0.0754 | +0.5 % | – |
| exp2_aug_s2 | 0.1169 | 0.1881 | 0.1903 | 0.0712 | 0.0734 | -1.0 % | – |
| exp4_aux | 0.1180 | 0.1856 | 0.1885 | 0.0676 | 0.0704 | -6.0 % | – |
| exp4_aux_s1 | 0.1174 | 0.1891 | 0.1915 | 0.0718 | 0.0741 | -0.2 % | – |
| exp4_aux_s2 | 0.1190 | 0.1883 | 0.1910 | 0.0694 | 0.0721 | -3.5 % | – |
| exp4_full | 0.1245 | 0.1749 | 0.1797 | 0.0504 | 0.0552 | -29.9 % | – |
| exp4_full_s1 | 0.1267 | 0.1800 | 0.1840 | 0.0533 | 0.0573 | -25.9 % | – |
| exp4_full_s2 | 0.1226 | 0.1761 | 0.1809 | 0.0534 | 0.0583 | -25.7 % | – |
| exp4_touch | 0.1249 | 0.1777 | 0.1834 | 0.0528 | 0.0585 | -26.6 % | – |
| exp4_touch_s1 | 0.1241 | 0.1762 | 0.1811 | 0.0521 | 0.0570 | -27.5 % | – |
| exp4_touch_s2 | 0.1221 | 0.1754 | 0.1809 | 0.0533 | 0.0589 | -25.8 % | – |
| **exp2_recipe** (family) | 0.1165 | 0.1884 | 0.1913 | 0.0719 | 0.0747 | +0.0 % | no |
| **exp4_aux** (family) | 0.1181 | 0.1877 | 0.1903 | 0.0696 | 0.0722 | -3.2 % | no |
| **exp4_full** (family) | 0.1246 | 0.1770 | 0.1815 | 0.0524 | 0.0569 | -27.2 % | no |
| **exp4_touch** (family) | 0.1237 | 0.1764 | 0.1818 | 0.0527 | 0.0581 | -26.6 % | no |

- Gap = held-out NLL − final training NLL. Held-out NLL covers the outer and pair predictions.
- Arm rows are single seeds; family rows average over seeds.
- The reference is exp_2's recipe, re-evaluated from exp_2's saved weights.
- **No family is flagged.** The flag fires on growth > 25 %; the touch families instead *shrink* the gap by ~27 %.
- **Touch features change the NLL in both directions.**
  - Training NLL rises from ~0.117 to ~0.124.
  - Held-out NLL falls from ~0.191 to ~0.182 (tl).
- **Cause untested [Low].** Candidates:
  - extra inputs slow fitting within fixed epochs;
  - the touch globals act as a regulariser.
- The aux head alone leaves the gap within 3 % of the reference.

## 8. Count gates

These are the exp_0 gates, applied against `baseline` on the same view and labels:

- gain ≥ 0.005;
- CI lower bound > 0;
- macro-F1 regression ≤ 0.01;
- every match's possession count within 30 %.

Each cell shows pass/fail and the worst |count error| with its match.

| variant | provider tl | provider orig | ctrl tl | ctrl orig |
|---|---|---|---|---|
| `exp3_ref_ens3` | fail: counts (0.37 J03WMX) | fail: counts (0.36 J03WMX) | pass (0.22 J03WR9) | pass (0.19 J03WR9) |
| `exp4_full_ens3` | fail: counts (0.34 J03WMX) | fail: counts (0.34 J03WMX) | pass (0.21 J03WOH) | pass (0.19 J03WR9) |
| `exp4_touch_ens3` | fail: counts (0.36 J03WMX) | fail: counts (0.35 J03WMX) | pass (0.18 J03WN1) | pass (0.18 J03WR9) |
| `exp4_aux_ens3` | fail: counts (0.33 J03WMX) | fail: counts (0.36 J03WMX) | pass (0.19 J03WOH) | pass (0.23 J03WR9) |
| `exp2_aug` | fail: counts (0.35 J03WMX) | fail: counts (0.35 J03WMX) | pass (0.22 J03WR9) | pass (0.22 J03WR9) |
| `exp4_full` | fail: counts (0.32 J03WMX) | fail: counts (0.30 J03WMX) | pass (0.20 J03WOH) | pass (0.17 J03WR9) |
| `exp4_full_s1` | fail: counts (0.32 J03WMX) | fail: counts (0.34 J03WMX) | pass (0.20 J03WR9) | pass (0.21 J03WR9) |
| `exp4_full_s2` | pass (0.30 J03WMX) | fail: counts (0.31 J03WN1) | pass (0.21 J03WN1) | pass (0.18 J03WR9) |
| `exp4_touch` | fail: counts (0.34 J03WMX) | fail: counts (0.33 J03WMX) | pass (0.17 J03WN1) | pass (0.20 J03WN1) |
| `exp4_touch_s1` | fail: counts (0.35 J03WMX) | fail: counts (0.32 J03WMX) | pass (0.19 J03WN1) | pass (0.18 J03WN1) |
| `exp4_touch_s2` | fail: counts (0.34 J03WMX) | fail: counts (0.33 J03WMX) | pass (0.19 J03WR9) | pass (0.21 J03WR9) |
| `exp4_aux` | fail: counts (0.35 J03WMX) | fail: counts (0.34 J03WMX) | pass (0.22 J03WR9) | pass (0.21 J03WR9) |
| `exp4_aux_s1` | fail: counts (0.34 J03WMX) | fail: counts (0.34 J03WMX) | pass (0.21 J03WR9) | pass (0.19 J03WOH) |
| `exp4_aux_s2` | fail: counts (0.37 J03WMX) | fail: counts (0.37 J03WMX) | pass (0.21 J03WR9) | pass (0.21 J03WR9) |
| `ryu_final` | fail: counts (0.52 J03WMX) | fail: counts (0.53 J03WMX) | pass (0.22 J03WOH) | pass (0.22 J03WOH) |
| `rasya_c08` | fail: macro,counts (0.62 J03WMX) | fail: counts (0.49 J03WMX) | fail: macro,counts (0.35 J03WMX) | pass (0.15 J03WOH) |

- **Provider labels.**
  - Every model arm fails only the count gate, including the reference.
  - The cause is almost always J03WMX, under-counted by about 30–37 %. It is a property of that match's provider labels relative to every decoder, not of exp_4.
  - The only pass is `exp4_full_s2` on tl, at 0.30, right at the 30 % limit. It does not count as an improvement.
- **L_ctrl.** Every exp_4 arm and the reference pass all gates. Only the context arm `rasya_c08` on tl fails.

**Aux target integrity** (`results/aux_integrity.csv`, notebook 00): 35 match × view/round arrays checked; max |aux ball − observed ball| on visible steps 0.00049 (float16 rounding); visible steps without a clean ball: 0; hidden steps with a ball target: 1.000 (min share); nearest-team target agrees with the provider possession team on 0.749–0.808 of alive steps (mean 0.773); arrays whose observed flip sign differs from the clean match: [('J03WN1', 'target_aug', '2', '13858')].

- The J03WN1 round-2 row is the case behind the pre-fit sign fix.
- The nearest-team target is a noisy proxy for possession, at about 77 % agreement on alive steps.

## 9. Limitations

- **Small evaluation set.**
  - There are 7 proxy matches, so the bootstrap resamples 7 units and "wins" are out of 7.
  - CIs of ±0.012 are as narrow as this design gets.
  - Every number here comes from the same 7 matches as exp_1–exp_3. Repeated comparisons on them make a chance "better" more likely over the series of experiments.
- **The proxy is not the target.**
  - `target_like` is a simulated degradation of IDSSE.
  - BAR-ATH transfer is untested: the public check is spent and no BAR-ATH data was read in exp_4.
- **The aux design was fixed a priori, not tuned.**
  - Fixed values: weights 0.25 / 0.25, smooth-L1 β 1.0, a linear head, nearest-player team.
  - H1 failing does not rule out other aux designs, such as a heavier weight, a ball-trajectory head or a ball-carrier target. Testing any of them needs a new pre-registration.
- **The aux team target is noisy.** The nearest player's team matches provider possession on only ~77 % of alive steps. The ball-xy target is exact (clean ball) but available only on IDSSE.
- **Seed noise is underestimated by three seeds.** The observed spread reached 0.019 and is itself a 3-sample range.
- **Single seeds only on decision labels.** They were not evaluated on `L_track(1 s)` / `L_track(3 s)`.
- **The training-NLL rise with touch features is unexplained (§7).**
- **Touch-only is post hoc.**
  - Its provider gain is the most interesting finding, but the protocol does not allow adopting it.
  - A confirmatory exp_5 would pre-register `exp4_touch_ens3` vs `exp3_ref_ens3` with fresh seeds. Fresh seeds address seed variance only; the 7 matches stay the same, so match-level selection risk remains.
  - It also does not help on `L_ctrl` (−0.003), which is the label set behind exp_3's primary BAR-ATH candidate.

## 10. Reproduction and hygiene

- **Notebooks.** `00_protocol` → `04_target` all executed via `run_notebook.py`. Each verifies the PREREG sha256 against `results/protocol.json` first.
- **Reproduction check.** One exp_2 fold re-trained through exp_4's loop, with aux off and no touch inputs, gives bit-exact predictions on both views (`results/repro_check.json`).
- **Reference recheck.** `exp3_ref_ens3` recomputed from cached predictions equals exp_3's stored rows (`results/reference_recheck.json`).
- **Fits.** 252 proxy fits plus 1 reproduction fit, out of a cap of 280. The full-data and rebuild fits were not needed. Total wall time for the fits was 14 492 s, about 57 s per fit.
- **Tests.** `tests/test_exp4.py` covers:
  - aux-off bit-exactness vs `exp1_net.train`;
  - RNG isolation of the aux head;
  - mirror swaps;
  - training-only aux targets;
  - the observed-round flip sign;
  - locks.
- **Data hygiene.**
  - Nothing under `data/` was written.
  - `evaluation/gt/` was not opened.
  - No BAR-ATH data was read: notebook 04 stopped at the failed-H1 branch.
  - exp_0–exp_3 were imported, never edited.

## Files

- `results/decision.json`: H1 rules, seed honesty, adoption and the report-only deltas.
- `results/proxy_report.json`: the full table, gates, gap table, short recall and reproduction check.
- `results/comparison_table.csv`, `results/per_match_results.csv`, `results/fig_exp4.png`.
- `results/gap_table.csv`, `train_nll.csv`, `heldout_nll.csv`, `ref_train_nll.csv`.
- `results/argmax_rows.csv`, `results/short_recall.csv`, `results/aux_integrity.csv`.
- `results/target_none.json`: no BAR-ATH files written.
