# exp_3 report — train on the competition's possession definition (pre-registered)

Date: 2026-11-09. Notebooks `00`–`06` in this folder are executed. Every number below comes from
`results/` (`label_stats_summary.json`, `sync_report.json`, `planning_rederivation.json`,
`proxy_report.json`, `decision.json`, `comparison_table.csv`, `per_match_results.csv`,
`coverage_rows.csv`, `s3_touch.json`, `target_fusion.json`, `target_plausibility.json`,
`target_candidate.json`).

## 1. Verdict

* **H1 failed.** Training on `L_ctrl` did not beat re-decoding the provider-trained ensemble. On
  `L_ctrl`, `target_like` view, Δ(`exp3_ctrl_ens3` − `exp3_ref_ens3`) = **−0.007 [−0.014, +0.002]**,
  with 2/7 wins. On the original view it is −0.010 [−0.018, +0.001]. The "partly mechanical" gain
  expected in PREREG §1 did not appear.
* **What moved the score was decoding on the target definition, not retraining.** With zero new fits,
  the cached exp_2 3-seed ensemble decoded with a config selected on `L_ctrl` reaches **0.748**
  (`target_like`). The same ensemble on provider labels is 0.664. `exp2_aug` decoded with exp_2's
  provider-selected configs scores 0.680 against the same `L_ctrl` labels. Re-selection mainly raises
  λ_team (final config λ_team 8, λ_dead 4, T 1.17, γ 0).
* **`L_ctrl` training is costly off-target.** The `L_ctrl`-trained ensemble loses 0.07 spellF1 on
  provider labels and 0.045 on `L_track(1 s)` against the provider-trained one, and it ties on
  `L_track(3 s)` (§4.2). A model trained on the finer provider labels plus a merging decoder covers
  more label definitions.
* **Adoption tree (pre-registered).** H1 failed, so the BAR-ATH candidate is `exp3_ref_ens3`: exp_2's
  three cached full-data models, averaged and decoded with the config selected on the 7
  `target_like` out-of-fold predictions against `L_ctrl` (§6).
* **Secondary.**
  * S2: extra ball coverage helps every arm (+0.007 to +0.017 spellF1).
  * S3: touch features give +0.012 [−0.004, +0.027] against `exp3_ctrl` seed 0, which is not
    significant.
  * On `L_ctrl`, every decoded SetSeq arm passes all exp_0 gates, including ±30 % counts. On
    provider labels, every decoded arm failed the count gate (exp_2).

## 2. What is new here vs exp_0 / exp_1 / exp_2 / rasya / ryu

| | exp_3 (this) | closest prior work |
|---|---|---|
| target definition | `L_ctrl`: CHALLENGE.md's controlled possession (short uncontrolled spells join the previous team) used for training, calibration, decoder selection and scoring | exp_0–exp_2: provider `BallPossession`; ryu: event-derived possessions (2.97/min) |
| label sensitivity | every arm re-selected and scored on provider, `L_track(1 / 2 / 3 s)` and `L_ctrl` | none |
| event-data sync | DFL events → tracking frames: kick-off sync + label-free snap, checked against the provider's own `CalculatedFrame` | none in the team |
| coverage sensitivity | clean ball re-revealed on 12 % of hidden frames, saved weights, fold configs fixed | none |
| fusion | exp_0's geometric acceptance (pinned verbatim, SHA-256) + a physical replay guard | exp_0: acceptance only |
| touch features | per-team min player–ball distance, peak ball-velocity change and its nearest team, from the 5 frames behind each step | exp_1: 5 fps snapshot geometry only |

## 3. Protocol and labels

* **Pre-registration.** `PREREG.md` SHA-256 `cff444ff…` was frozen at 2026-11-08T16:53:50Z, before
  any exp_3 fit and before any model was scored on `L_ctrl`. Every notebook re-checks the hash, the
  judge, the fusion pin and all constants read from the code. PREREG §0 lists what was seen before
  freezing; the experiment is not blind.
* **Judge and protocol.** exp_1's frozen judge, round-0 held-out corruption, and exp_2's views. Nested
  leave-one-match-out: 21 pair + 7 outer fits per seed. Temperature and decoder (`D.GAMMAS` ×
  λ grid) are selected on the six inner validation matches with exp_0's objective, *on the
  evaluation label set of each analysis*. Official `possession_metrics.py`; paired match bootstrap,
  1000 draws, seed 20261108.
* **Fits.**
  * 84 `exp3_ctrl` (3 seeds × 28) + 28 `exp3_touch` + full-data / rebuild fits (§6).
  * Fits took 45–52 s each.
  * The exp_2 caches were read in place with their fingerprints checked; no exp_2 arm was re-fitted.

### 3.1 Event sync and the fallback

Rule (c) of `L_ctrl` needs controlled on-ball events located to within ±0.5 s. The open IDSSE event
files (Figshare 10.6084/m9.figshare.28196177, CC BY 4.0) were parsed with lxml (schema in
`DATA_SOURCES.md`).

| sync (pooled over 7 matches, shots with provider `CalculatedFrame`) | shots | median \|r\| (frames) | p90 \|r\| (frames) | accepted |
|---|---|---|---|---|
| kick-off only (`EventTime` − kick-off) | 159 | 30 | 99 | no |
| kick-off + label-free snap (player's closest approach to the ball in [−5 s, +2 s]) | 152 | 2 | 57.5 | **no** (criterion p90 ≤ 12) |

* Event times are logged late, by −3 s to +10 s. The snap fixes most events but leaves a 10–20 %
  tail on the wrong touch.
* Under the pre-registered rule the events are **not used**. `L_ctrl` = rules (a) + (b), which is
  identical to `L_track(2 s)` on all 7 matches (`label_integrity.json`).
* Against provider possession, the snapped events agree with the owning team at lag 0 on 88–91 % of
  events, and 95 % at the best lag (2–19 frames).

### 3.2 Label-only statistics (mean of 7 matches; reported before any model was scored on `L_ctrl`)

| label set | possessions / min | mean possession (s) | frames changed vs provider | agreement with `L_ctrl` |
|---|---|---|---|---|
| provider | 4.58 | 7.3 | – | 0.983 |
| `L_track(1 s)` | 3.83 | 8.7 | 0.45 % | 0.986 |
| **`L_ctrl`** (= `L_track(2 s)`) | **3.02** | 11.0 | 1.71 % | 1 |
| `L_track(3 s)` | 2.49 | 13.4 | 3.11 % | 0.983 |

Dead share is 0.449 in every set, since D is never changed.

### 3.3 Planning findings, re-derived (cached `exp2_aug`, `target_like`, exp_2's own fold configs)

| finding | planning session | re-derived here |
|---|---|---|
| 5 fps quantisation ceiling | 0.973 | 0.973 |
| model / oracle dead-alive / oracle team (both alive) | – / 0.679 / 0.944 | 0.657 / 0.677 / **0.927** |
| missed possessions (IoU 0.5) under 5 s | 86 % | 86.6 % (1210 of 3061) |
| possessions under 2 s: share / recall decoded / recall argmax | 30 % / 0.28 / ≈0.32 | 30.0 % / 0.284 / 0.321 |
| misses with ball visible on ≥ 80 % of frames | 42 % | 42.2 % |
| mean p(true team), < 2 s possessions with the ball visible | 0.58 | **0.60** |
| `exp2_aug` against `L_track(1 / 2 s)` (no retraining, no re-selection) | 0.693 / 0.683 | 0.692 / 0.680 (count error −5 % / +21 %) |

Two values differ, shown in bold. My oracle-team definition replaces the team on 5 fps steps where
truth and prediction are both alive. The planning script is not in the repo, so its exact
definitions are unknown [Medium].

## 4. Proxy results (7 held-out degraded IDSSE matches)

### 4.1 Primary: `L_ctrl` labels

**`target_like` view (primary)**

| arm | mean spellF1 | min | macroF1 | mean count err / max \|count err\| | Δ vs `exp3_ref_ens3` [95% CI] | Δ vs `exp2_aug` [95% CI] | ±30 % count gate |
|---|---|---|---|---|---|---|---|
| `exp3_ctrl_ens3` | 0.742 | 0.685 | 0.921 | -0.144 / 0.187 | -0.007 [-0.014, +0.002] | +0.007 [-0.007, +0.021] | pass |
| **`exp3_ref_ens3`** | **0.748** | 0.706 | 0.923 | -0.125 / 0.225 | – | +0.014 [-0.002, +0.026] | pass |
| `exp3_ctrl` (seed 0) | 0.725 | 0.655 | 0.918 | -0.158 / 0.194 | -0.023 [-0.033, -0.015] | -0.009 [-0.029, +0.008] | pass |
| `exp3_ctrl_s1` | 0.718 | 0.687 | 0.917 | -0.109 / 0.158 | -0.030 [-0.041, -0.021] | -0.016 [-0.030, -0.002] | pass |
| `exp3_ctrl_s2` | 0.728 | 0.683 | 0.917 | -0.128 / 0.215 | -0.020 [-0.031, -0.009] | -0.006 [-0.025, +0.011] | pass |
| `exp2_aug` | 0.735 | 0.705 | 0.919 | -0.105 / 0.221 | -0.014 [-0.026, +0.002] | – | pass |
| `exp1_ref` | 0.729 | 0.694 | 0.914 | -0.117 / 0.190 | -0.019 [-0.030, -0.006] | -0.005 [-0.019, +0.012] | pass |
| `exp3_ctrl_ens3_argmax` (diagnostic) | 0.698 | 0.626 | 0.921 | +0.053 / 0.123 | -0.050 [-0.062, -0.041] | -0.037 [-0.055, -0.020] | pass |
| `exp3_ref_ens3_argmax` (diagnostic) | 0.631 | 0.576 | 0.919 | +0.402 / 0.514 | -0.117 [-0.123, -0.111] | -0.103 [-0.121, -0.087] | FAIL |
| `ryu_final` (zero-shot) | 0.619 | 0.557 | 0.878 | -0.166 / 0.215 | -0.129 [-0.156, -0.103] | -0.115 [-0.136, -0.094] | pass |
| `rasya_c08` | 0.304 | 0.272 | 0.375 | -0.286 / 0.352 | -0.444 [-0.463, -0.421] | -0.430 [-0.446, -0.411] | FAIL |
| `baseline` | 0.219 | 0.200 | 0.615 | +0.857 / 1.055 | -0.529 [-0.543, -0.514] | -0.516 [-0.536, -0.499] | FAIL |

**original view**

| arm | mean spellF1 | min | macroF1 | mean count err / max \|count err\| | Δ vs `exp3_ref_ens3` [95% CI] | Δ vs `exp2_aug` [95% CI] | ±30 % count gate |
|---|---|---|---|---|---|---|---|
| `exp3_ctrl_ens3` | 0.741 | 0.689 | 0.922 | -0.134 / 0.175 | -0.010 [-0.018, +0.001] | +0.011 [-0.003, +0.026] | pass |
| **`exp3_ref_ens3`** | **0.751** | 0.707 | 0.925 | -0.113 / 0.195 | – | +0.021 [+0.007, +0.035] | pass |
| `exp3_ctrl` (seed 0) | 0.728 | 0.661 | 0.920 | -0.154 / 0.183 | -0.022 [-0.031, -0.015] | -0.002 [-0.021, +0.015] | pass |
| `exp3_ctrl_s1` | 0.722 | 0.680 | 0.918 | -0.123 / 0.197 | -0.029 [-0.043, -0.013] | -0.008 [-0.024, +0.010] | pass |
| `exp3_ctrl_s2` | 0.730 | 0.689 | 0.917 | -0.124 / 0.190 | -0.021 [-0.035, -0.007] | -0.000 [-0.016, +0.016] | pass |
| `exp2_aug` | 0.730 | 0.690 | 0.917 | -0.106 / 0.217 | -0.021 [-0.035, -0.007] | – | pass |
| `exp1_ref` | 0.735 | 0.702 | 0.920 | -0.096 / 0.176 | -0.015 [-0.029, -0.003] | +0.006 [-0.010, +0.024] | pass |
| `exp3_ctrl_ens3_argmax` (diagnostic) | 0.699 | 0.627 | 0.923 | +0.055 / 0.116 | -0.052 [-0.063, -0.039] | -0.031 [-0.053, -0.010] | pass |
| `exp3_ref_ens3_argmax` (diagnostic) | 0.635 | 0.588 | 0.920 | +0.397 / 0.546 | -0.115 [-0.124, -0.105] | -0.094 [-0.112, -0.078] | FAIL |
| `ryu_final` (zero-shot) | 0.629 | 0.548 | 0.887 | -0.182 / 0.222 | -0.122 [-0.157, -0.089] | -0.101 [-0.121, -0.081] | pass |
| `rasya_c08` | 0.365 | 0.338 | 0.592 | -0.113 / 0.154 | -0.386 [-0.409, -0.361] | -0.365 [-0.382, -0.343] | pass |
| `baseline` | 0.198 | 0.182 | 0.459 | +0.776 / 0.971 | -0.552 [-0.569, -0.532] | -0.531 [-0.547, -0.515] | FAIL |

Seed spread of the three `exp3_ctrl` seeds on `L_ctrl`: 0.010 (`target_like`), 0.008 (original).
Ensembling adds +0.018 (`target_like`) / +0.014 (original) over the seed mean.

**Per match, `L_ctrl`, `target_like`** (spellF1, count error in brackets)

| match | `exp3_ctrl_ens3` | `exp3_ref_ens3` | `exp2_aug` | `exp3_ctrl` | `ryu_final` |
|---|---|---|---|---|---|
| J03WMX | 0.754 (-0.17) | 0.748 (-0.14) | 0.719 (-0.11) | 0.727 (-0.18) | 0.595 (-0.18) |
| J03WN1 | 0.722 (-0.19) | 0.740 (-0.13) | 0.710 (-0.01) | 0.722 (-0.18) | 0.557 (-0.20) |
| J03WOH | 0.765 (-0.15) | 0.778 (-0.19) | 0.758 (-0.14) | 0.748 (-0.19) | 0.665 (-0.22) |
| J03WOY | 0.752 (-0.12) | 0.753 (-0.08) | 0.724 (-0.10) | 0.739 (-0.17) | 0.586 (-0.20) |
| J03WPY | 0.768 (-0.14) | 0.758 (-0.07) | 0.785 (-0.09) | 0.742 (-0.13) | 0.643 (-0.10) |
| J03WQQ | 0.746 (-0.09) | 0.755 (-0.04) | 0.740 (-0.06) | 0.744 (-0.09) | 0.648 (-0.15) |
| J03WR9 | 0.685 (-0.13) | 0.706 (-0.22) | 0.705 (-0.22) | 0.655 (-0.16) | 0.640 (-0.12) |

**Why retraining did not help.** The argmax diagnostic shows what the labels change.

* The `L_ctrl`-trained frame model is much smoother on its own: argmax count error is +5 %, against
  +40 % for the provider-trained one.
* But a high-λ_team Viterbi does the same merging on the provider model's probabilities, while the
  finer provider training keeps more information about where possession really changes.
* After decoding, both under-segment by 12–14 %. The provider-trained ensemble keeps the edge.

### 4.2 S1 — label sensitivity (mean spellF1; each cell re-selected on that label set)

| arm | provider orig / TL | `L_track(1 s)` orig / TL | **`L_ctrl`** orig / TL | `L_track(3 s)` orig / TL |
|---|---|---|---|---|
| `exp3_ref_ens3` | 0.666 / 0.664 | 0.707 / 0.709 | **0.751 / 0.748** | 0.759 / 0.762 |
| `exp3_ctrl_ens3` | 0.595 / 0.594 | 0.664 / 0.662 | 0.741 / 0.742 | 0.760 / 0.762 |
| `exp3_ctrl` (seed 0) | 0.585 / 0.584 | 0.651 / 0.649 | 0.728 / 0.725 | 0.755 / 0.753 |
| `exp2_aug` | 0.657 / 0.657 | 0.702 / 0.698 | 0.730 / 0.735 | 0.749 / 0.752 |
| `exp1_ref` | 0.656 / 0.648 | 0.698 / 0.685 | 0.735 / 0.729 | 0.749 / 0.744 |
| `ryu_final` (zero-shot) | 0.487 / 0.479 | 0.548 / 0.539 | 0.629 / 0.619 | 0.666 / 0.656 |
| `baseline` | 0.186 / 0.204 | 0.198 / 0.218 | 0.198 / 0.219 | 0.183 / 0.203 |

Δ(`exp3_ctrl_ens3` − `exp3_ref_ens3`), target_like:

| label set | Δ [95% CI] | wins |
|---|---|---|
| provider | −0.071 [−0.079, −0.063] | 0/7 |
| `L_track(1 s)` | −0.047 [−0.054, −0.040] | 0/7 |
| `L_ctrl` | −0.007 [−0.014, +0.002] | 2/7 |
| `L_track(3 s)` | +0.000 [−0.012, +0.016] | 3/7 |

* **The ranking is stable.** The provider-trained ensemble with re-selection is best or tied under
  every definition.
* **Count bias** (`target_like`, mean count error of `exp3_ref_ens3`): provider −0.20, `L_track(1 s)`
  −0.14, `L_ctrl` −0.13, `L_track(3 s)` −0.04. Re-selection on the evaluation labels removes most of
  the over-segmentation that the label change would otherwise cause.
* **The absolute level is set by the label definition** (0.66 → 0.76), not by the model. That is the
  real risk for BAR-ATH (§7).

### 4.3 S2 — coverage sensitivity (clean ball re-revealed; saved weights; fold configs from `target_like`)

The view reveals 12.0 % of hidden frames, raising ball visibility from 0.395 to 0.468 (mean of 7
matches).

| arm | `L_ctrl`: Δ [95% CI], wins | provider: Δ [95% CI], wins |
|---|---|---|
| `exp3_ref_ens3` | +0.007 [+0.000, +0.014], 5/7 | +0.010 [+0.002, +0.017], 6/7 |
| `exp3_ctrl_ens3` | +0.012 [+0.007, +0.017], 7/7 | +0.010 [+0.004, +0.016], 6/7 |
| `exp3_ctrl` | +0.017 [+0.010, +0.024], 6/7 | +0.008 [+0.001, +0.014], 5/7 |
| `exp2_aug` | +0.012 [+0.004, +0.019], 6/7 | +0.013 [+0.006, +0.018], 6/7 |

More ball helps every arm by about 0.01 with no retraining. If broadcast fusion recovers a
similar share of BAR-ATH's missing ball cleanly, it is worth using. This is the motivation for
§6's fusion rule. The proxy reveals the *clean* ball, so it is an optimistic bound on what noisy
broadcast positions can give.

### 4.4 S3 — touch features (stretch; report only; 28 fits; seed 20261108)

| labels / view | `exp3_touch` − `exp3_ctrl` (seed 0) [95% CI] | wins |
|---|---|---|
| `L_ctrl` / target_like | +0.012 [−0.004, +0.027] | 6/7 |
| `L_ctrl` / original | +0.012 [−0.004, +0.026] | 6/7 |
| provider / target_like | +0.013 [−0.001, +0.030] | 5/7 |
| provider / original | +0.011 [−0.003, +0.027] | 5/7 |

* The effect is consistently positive and as large as the seed spread (0.010), but the CI includes 0
  everywhere. One seed only; not adopted (PREREG: S3 never changes the candidate).
* The 4 features swap correctly under the home/away mirror (unit test). The regenerated training
  observations matched the cached tokens' ball visibility on every match and round.
* Worth a 3-seed follow-up on provider labels, where the gap is largest.

## 5. Pre-registered decisions (PREREG §6, `results/decision.json`)

| rule | statistic | result |
|---|---|---|
| H1 — `exp3_ctrl_ens3` − `exp3_ref_ens3`, `L_ctrl`, `target_like`, CI lower > 0 | −0.0065 [−0.0140, +0.0018], 2/7 wins | **failed** |
| Adoption tree | H1 failed → `exp3_ref_ens3` | **`exp3_ref_ens3`**, final config T 1.170, λ_team 8, λ_dead 4, γ 0 |
| Event use for `L_ctrl` rule (c) — shot residual median ≤ 5 and p90 ≤ 12 frames | median 2, p90 57.5 | **failed → fallback (a) + (b)** |
| exp_0 gates on `L_ctrl` (report only): gain ≥ 0.05, CI > 0, macroF1, ±30 % counts | every decoded SetSeq arm, both views | **all pass** (worst count error 0.225, `exp3_ref_ens3`, J03WR9) |

The selected config for `exp3_ctrl_ens3` (not adopted) was T 1.20, λ_team 4, λ_dead 1, γ 0.5. γ > 0
was chosen for the first time in exp_1–exp_3. On `L_ctrl` this happened in some folds for `exp1_ref`
(original) and `exp2_aug` (both views), and in the OOF selection for `exp3_ctrl_ens3`. The adopted
`exp3_ref_ens3` uses γ = 0 in every fold.

## 6. BAR-ATH (label-free aggregates only)

Sources: `results/target_fusion.json`, `target_plausibility.json` and `target_candidate.json`. These
numbers are report-only and never used for a choice. BAR-ATH was read only in notebook 05's kernel.

**Broadcast fusion (pinned exp_0 rule + replay guard): rejected in both halves → tactical input only.**

| half | offset (frames) | overlap frames | median distance (m) | p90 distance (m) | accepted (median ≤ 1.5, p90 ≤ 5) |
|---|---|---|---|---|---|
| 1 | +4014 | 38 449 | 0.79 | 6.71 | no |
| 2 | +26 920 | 35 752 | 0.84 | 6.31 | no |

* The broadcast and tactical balls agree closely in the median but have a heavy tail (p90 > 5 m).
  The pre-registered rule therefore refuses the broadcast ball.
* No frame was recovered, so the replay guard had nothing to filter. Graded ball coverage stays
  0.593, with 64 643 graded frames missing the ball.
* S2 suggests about +0.01 spellF1 is left on the table. A per-frame acceptance rule (reject only
  disagreeing frames) would be the next fusion design. It is not done here.

**Plausibility (label-free; decoders selected on the 7 `target_like` OOF predictions against `L_ctrl`).**

| pipeline | dead share | possessions / min | mean possession (s) | possessions |
|---|---|---|---|---|
| exp_1 shipped (`exp_1_setseq_bd.csv`) | 0.384 | 4.13 | 9.0 | 437 |
| exp_2 candidate (`exp2_aug`, provider-selected) | 0.408 | 3.57 | 10.0 | 378 |
| `exp3_ctrl_ens3` (not adopted) | 0.417 | 2.41 | 14.5 | 255 |
| **`exp3_ref_ens3` (adopted candidate)** | **0.410** | **2.45** | **14.5** | **259** |
| references | prior 0.407 | ryu 2.97; proxy `L_ctrl` truth 3.02 | – | – |

**Candidate.**

* File: `predictions/exp_3_candidate.csv` (gitignored). Variant `exp3_ref_ens3`, the mean of exp_2's
  cached full-data `exp2_aug`, `_s1` and `_s2` models.
* Input: tactical only. Decoder: T 1.170, λ_team 8, λ_dead 4, γ 0.
* The file passes the official validator. SHA-256 `17eefbcc…7c08`. No graded frame abstains.
* Three **independent** re-fits of the members give a **byte-identical** file. The public ground
  truth was not opened.
* **Reading the counts [Medium].**
  * The candidate predicts 2.45 possessions/min. That is 31 % below exp_2's file and below both ryu
    (2.97) and the proxy `L_ctrl` truth rate (3.02).
  * On the proxy, the same arm under-segments `L_ctrl` by about 12 %. A BAR-ATH truth near
    3.0/min would therefore be expected to give about 2.6/min, which fits.
  * If BAR-ATH's ground truth follows the provider-like definition (about 4.6/min), this file
    under-counts by about half and would score far below exp_2's.
  * The dead share sits on the prior (0.410 vs 0.407), as exp_2's did.
* **Do not submit this file without the team deciding the label definition (§7, §8).**

### 6.1 Post-hoc hedge (NOT pre-registered; notebook 07, `results/target_hedge_provider.json`)

Because the label definition is the dominant unknown (§7), a second file was built from the **same**
cached ensemble. The only difference from the candidate is the decoder, selected on the 7
`target_like` out-of-fold predictions against **provider** labels instead of `L_ctrl`. It does not
replace the pre-registered candidate.

| file | decoder (T / λ_team / λ_dead / γ) | dead share | possessions / min | mean possession (s) | possessions | SHA-256 | rebuild |
|---|---|---|---|---|---|---|---|
| `exp_3_candidate.csv` (`L_ctrl`-selected, adopted) | 1.17 / 8 / 4 / 0 | 0.410 | 2.45 | 14.5 | 259 | `17eefbcc…7c08` | byte-identical (3 fits) |
| `exp_3_hedge_provider.csv` (provider-selected) | 1.15 / 0.5 / 4 / 0 | 0.408 | 3.75 | 9.5 | 397 | `a6ebb94b…96f3` | byte-identical (3 fits) |

* Both files pass the official validator and abstain on no graded frame. Neither used the public
  ground truth.
* Proxy expectation from S1 (`target_like`):
  * The hedge's arm scores 0.664 on provider labels.
  * The candidate's arm scores 0.748 on `L_ctrl`.
  * Each is the better file if its definition is BAR-ATH's.
* The two differ only in λ_team (8 vs 0.5). That one setting, chosen by the label definition, moves
  BAR-ATH's predicted count by 53 %.

**Fit budget used.** 84 (`exp3_ctrl`) + 28 (`exp3_touch`) + 3 full-data `exp3_ctrl` (plausibility
only) + 3 + 3 independent rebuild fits (candidate, hedge) = **121** of the 150 cap. No exp_2 arm was
re-fitted except for the rebuild fits.

## 7. Limitations and threats to validity

* **`L_ctrl` vs BAR-ATH's ground truth is untested and untestable here.** CHALLENGE.md says the ground
  truth "takes the team the tracking names as owning it", derived mechanically from corrected
  tracking. That is the same *kind* of field as IDSSE's `BallPossession`.
  * If BAR-ATH's owner field behaves like IDSSE's (4.6/min), the provider definition is the right
    proxy. The adopted candidate is decoded for `L_ctrl`, and the same ensemble re-selected on
    provider labels would be the better choice (S1: 0.664 vs that candidate's unknown provider-label
    score).
  * If it behaves like controlled possession (≈ 3/min), the candidate is right.
  * CHALLENGE.md's own text mentions 17 possessions in the 5-minute public window (3.4/min), between
    `L_track(1 s)` and `L_ctrl`. It was not used for any choice, and one 5-minute window is weak
    evidence.
* **The event route failed on sync, not on the idea.** Rule (c) was never tested. A better sync (e.g.
  matching event positions to tracking, as in ETSY / databallpy) could still give a definition
  between `L_track(1 s)` and `L_ctrl`.
* **Not blind.** Under the fallback, `L_ctrl` equals `L_track(2 s)`, whose statistics and one
  `exp2_aug` score (0.683 / 0.680) were seen in planning before freezing (PREREG §0).
* **Selection is the main effect.** The gain over exp_2 (0.68 → 0.75 on `L_ctrl`) comes from choosing
  T/λ on the target labels. Any label-definition error on BAR-ATH goes straight into that choice.
* **Seven matches, one league, one provider.** The paired CIs are wide (±0.008–0.015), and S3's
  positive trend is not established.
* **S2 is an optimistic proxy** (clean ball). Broadcast positions are noisier and can include
  replays; hence the replay guard.
* **Runtime incident.** Two session interruptions killed notebooks 03 and 06 mid-run. Both were
  re-run; 03 resumed from the result files already written by the same function (no partial file was
  reused). The cached-arm S1 evaluations were precomputed by a helper script calling the same
  `evaluate_variant` while the fits ran.

## 8. Next steps (not done here)

1. **Decide the target definition with the team** (provider-like vs controlled), because it moves the
   decoder by more than any model change. The cheapest hedge is two candidates from the same cached
   ensemble: one selected on provider labels, one on `L_ctrl`.
2. **Event sync done properly** (position matching), to test rule (c) and a definition between 1 s
   and 2 s.
3. **S3 with 3 seeds**, and auxiliary hidden-ball supervision (BallRadar-style). Spells in ball gaps
   are about half the misses.
4. **SkillCorner leave-one-source-out** with harmonised labels; integrate into
   `scripts/build_submission.py`.
