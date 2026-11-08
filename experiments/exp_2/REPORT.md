# exp_2 report — transfer-robust SetSeq (pre-registered)

Date: 2026-11-08. Notebooks `00`–`05` in this folder are executed. Every number below is copied from
`results/` (`proxy_report.json`, `decision.json`, `comparison_table.csv`, `per_match_results.csv`,
`view_stats.csv`, `reproduction_exp1_ref.json`, `target_plausibility.json`, `target_candidate.json`).
The tables in §4 were rendered from those files by `make_report_tables.py` (`results/report_tables.md`).

## 1. Verdict

* **H1 is supported by the pre-registered rules, with a small margin.** Training SetSeq with target-like
  observation augmentation (`exp2_aug`) improves spellF1 on the `target_like` view by **+0.009
  [+0.001, +0.018]** (5 of 7 matches). On the original view the change is +0.001 [−0.008, +0.010],
  within the 0.02 tolerance. The CI lower bound sits just above zero. The two extra training seeds of
  the same recipe give +0.001 and +0.008 against `exp1_ref`, so the seed-0 result is one favourable draw.
* **The feared transfer failure is small for SetSeq.** Under `target_like`, exp_1's model loses only
  −0.008 [−0.012, −0.004]. rasya's rules lose −0.06, because they read the off-pitch ball as a
  dead-ball cue. ryu loses −0.009. `exp2_aug` removes the sensitivity entirely: −0.000 [−0.005, +0.005].
* **H2 (count-aware decoding) is not supported.** The explicit-duration decoder is non-inferior
  (−0.002 [−0.006, +0.001]), but J03WMX stays at −34 %, so the ±30 % count gate still fails. Every
  decoded arm fails that gate on that same match.
* **H3 (3-seed ensemble) passes nominally, by 7×10⁻⁶.** |Δ| is 0.007490 against a seed spread of
  0.007483, and the CI is [+0.001, +0.014]. The margin is noise-level, so this is a nominal pass,
  not a demonstrated effect. The ensemble is the best arm on both views (0.664 / 0.666, 7 of 7 wins
  over `exp1_ref` on `target_like`), but it was not the pre-registered candidate.
* **Adoption rule fired.** `predictions/exp_2_candidate.csv` (the `exp2_aug` arm) passes the official
  validator and rebuilds byte-identically from an independent fit. On BAR-ATH its dead share (0.408)
  matches the published prior (0.407), but it predicts fewer possessions than exp_1 (378 vs 437);
  see §6. Whether to submit it instead of exp_1's file is a team decision. The proxy gain is small and
  the BAR-ATH count moved down.

## 2. What is new here vs exp_0 / exp_1 / rasya / ryu

| | exp_2 (this) | closest prior work |
|---|---|---|
| evaluation | second, label-free **`target_like` view** of the same 7 held-out matches, applied to every arm | exp_0/exp_1: one in-domain view; ryu: other sources, zero-shot |
| training | **`target_aug`**: rounds 1–2 lose off-pitch ball tails (random length) and, in half of the long dead spells, the ball | exp_1: corruption resampling + mirror |
| decoder | **explicit-duration HSMM**, entropy-centred log-normal duration priors from the training folds' labels | exp_1: constant / boundary Viterbi; ryu: rate-calibrated λ |
| ensemble | 3 training seeds, probabilities averaged | ryu: 3-seed TCN |
| reproducibility | exp_1 re-fit is **bit-exact** (max abs probability difference 0.0 on all 7 outer folds) | – |

## 3. Protocol

* **Pre-registration.** `PREREG.md` SHA-256 `154a10f7cda6821eb7322801fc1788d6170bcf81680e5d7164ba11564a0acb52`
  was frozen at 2026-11-08T13:06:55Z, before any exp_2 fit. `freeze_prereg()` refuses to run once any
  prediction exists. Every notebook re-checks the hash, the judge hashes and every constant, read
  from the code. PREREG §0 lists what was looked at before freezing, including one decoder smoke test
  that led to entropy-centring the duration term.
* **Judge.** This is exp_1's frozen snapshot, hash-checked. The round-0 copies are bit-exact with the
  degrader (`results/view_integrity.json`).
* **Nesting.** This is exp_1's protocol. It uses 21 pair models and 7 outer models per arm, which
  makes 4 arms × 28 = 112 proxy fits, plus 3 full-data fits and 1 rebuild fit, for 116 in total.
  Each fit predicted both views with the same weights. Decoder settings are selected per view on the
  six inner validation matches with exp_0's objective. The duration priors use only the decoded
  model's training labels. An independent review checked this and found no leakage.
* **Statistics.** Paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval, official
  `possession_metrics.py`.

**What the views do** (`results/view_stats.csv`, mean of 7 matches; labels only describe):

| view | ball visible | off-pitch of visible | ball visible on dead frames | on alive frames |
|---|---|---|---|---|
| original (round 0) | 0.590 | 0.264 | 0.581 | 0.597 |
| `target_like` (held-out) | 0.395 | 0.040 | 0.151 | 0.595 |
| `target_aug` (training rounds 1–2) | 0.394 | 0.034 | 0.147 | 0.596 |

BAR-ATH: ball visible 0.593, off-pitch 4.2 % of visible. `target_like` matches the off-pitch share but
is harsher on coverage. It also assumes that a missing ball goes with dead time, which is unverified
for BAR-ATH.

## 4. Proxy results (7 held-out degraded IDSSE matches)

**original view**

| arm | mean spellF1 | min | macroF1 | mean / max abs count err | Δ vs exp1_ref [95% CI] | wins | count gate |
|---|---|---|---|---|---|---|---|
| `exp1_ref` | 0.656 | 0.616 | 0.918 | 0.212 / 0.342 | – | – | FAIL |
| `exp2_aug` | 0.657 | 0.596 | 0.917 | 0.200 / 0.354 | +0.001 [-0.008, +0.010] | 3/7 | FAIL |
| `exp2_aug_const` | 0.657 | 0.596 | 0.917 | 0.200 / 0.354 | +0.001 [-0.008, +0.010] | 3/7 | FAIL |
| `exp2_aug_hsmm` | 0.656 | 0.599 | 0.917 | 0.190 / 0.337 | +0.001 [-0.009, +0.010] | 4/7 | FAIL |
| `exp2_aug_s1` | 0.651 | 0.617 | 0.918 | 0.192 / 0.344 | -0.005 [-0.016, +0.006] | 3/7 | FAIL |
| `exp2_aug_s2` | 0.656 | 0.618 | 0.917 | 0.220 / 0.335 | +0.001 [-0.008, +0.011] | 3/7 | FAIL |
| `exp2_aug_ens3` | 0.666 | 0.621 | 0.922 | 0.204 / 0.360 | +0.011 [+0.004, +0.019] | 6/7 | FAIL |
| `exp1_ref_argmax` (diagnostic) | 0.624 | 0.557 | 0.918 | 0.089 / 0.182 | -0.032 [-0.043, -0.022] | 0/7 | pass |
| `exp2_aug_argmax` (diagnostic) | 0.628 | 0.552 | 0.917 | 0.082 / 0.209 | -0.027 [-0.043, -0.014] | 0/7 | pass |
| `ryu_final` (zero-shot) | 0.487 | 0.404 | 0.879 | 0.457 / 0.528 | -0.168 [-0.202, -0.132] | 0/7 | FAIL |
| `rasya_c08` | 0.325 | 0.308 | 0.591 | 0.411 / 0.489 | -0.331 [-0.351, -0.309] | 0/7 | FAIL |
| `rasya_c06` | 0.322 | 0.301 | 0.634 | 0.357 / 0.464 | -0.333 [-0.349, -0.315] | 0/7 | FAIL |
| `baseline` | 0.186 | 0.167 | 0.458 | 0.179 / 0.354 | -0.469 [-0.487, -0.450] | 0/7 | FAIL |

**`target_like` view (primary)**

| arm | mean spellF1 | min | macroF1 | mean / max abs count err | Δ vs exp1_ref [95% CI] | wins | count gate |
|---|---|---|---|---|---|---|---|
| `exp1_ref` | 0.648 | 0.598 | 0.913 | 0.204 / 0.342 | – | – | FAIL |
| `exp2_aug` | 0.657 | 0.604 | 0.917 | 0.199 / 0.354 | +0.009 [+0.001, +0.018] | 5/7 | FAIL |
| `exp2_aug_const` | 0.657 | 0.604 | 0.917 | 0.199 / 0.354 | +0.009 [+0.001, +0.018] | 5/7 | FAIL |
| `exp2_aug_hsmm` | 0.654 | 0.592 | 0.917 | 0.185 / 0.341 | +0.006 [-0.003, +0.016] | 5/7 | FAIL |
| `exp2_aug_s1` | 0.649 | 0.610 | 0.917 | 0.192 / 0.352 | +0.001 [-0.008, +0.010] | 4/7 | FAIL |
| `exp2_aug_s2` | 0.656 | 0.607 | 0.917 | 0.223 / 0.337 | +0.008 [-0.002, +0.019] | 5/7 | FAIL |
| `exp2_aug_ens3` | 0.664 | 0.615 | 0.921 | 0.202 / 0.366 | +0.016 [+0.010, +0.023] | 7/7 | FAIL |
| `exp1_ref_argmax` (diagnostic) | 0.616 | 0.542 | 0.912 | 0.086 / 0.170 | -0.032 [-0.043, -0.022] | 0/7 | pass |
| `exp2_aug_argmax` (diagnostic) | 0.627 | 0.550 | 0.917 | 0.085 / 0.223 | -0.021 [-0.036, -0.007] | 2/7 | pass |
| `ryu_final` (zero-shot) | 0.479 | 0.401 | 0.870 | 0.446 / 0.523 | -0.169 [-0.203, -0.133] | 0/7 | FAIL |
| `rasya_c08` | 0.266 | 0.232 | 0.374 | 0.524 / 0.622 | -0.382 [-0.403, -0.353] | 0/7 | FAIL |
| `rasya_c06` | 0.260 | 0.226 | 0.403 | 0.499 / 0.607 | -0.388 [-0.406, -0.364] | 0/7 | FAIL |
| `baseline` | 0.204 | 0.182 | 0.614 | 0.234 / 0.431 | -0.443 [-0.466, -0.421] | 0/7 | FAIL |

`exp1_ref` on the original view reproduces exp_1's 0.656 exactly; its predictions are bit-identical
to exp_1's cache. On both views, nested selection chose γ = 0 in every fold for every SetSeq arm, so
`exp2_aug` ≡ `exp2_aug_const`, just as in exp_1.

**Robustness: `target_like` − `original`, paired by match**: `exp1_ref` −0.008 [−0.012, −0.004];
`exp2_aug` −0.000 [−0.005, +0.005]; `exp2_aug_ens3` −0.002 [−0.005, +0.001]; `ryu_final` −0.009
[−0.014, −0.004]; `rasya_c06` −0.062 [−0.072, −0.053]; `rasya_c08` −0.059 [−0.073, −0.045];
`baseline` +0.018 [+0.014, +0.024]. The baseline gain was not diagnosed. A plausible cause is that
hiding the parked ball removes nearest-player votes during dead time.

**Per match, `target_like` (spellF1, count error in brackets)**

| match | `exp1_ref` | `exp2_aug` | `exp2_aug_hsmm` | `exp2_aug_ens3` | `ryu_final` |
|---|---|---|---|---|---|
| J03WMX | 0.604 (-0.34) | 0.621 (-0.35) | 0.622 (-0.34) | 0.615 (-0.37) | 0.429 (-0.52) |
| J03WN1 | 0.639 (-0.27) | 0.639 (-0.24) | 0.640 (-0.25) | 0.659 (-0.29) | 0.401 (-0.50) |
| J03WOH | 0.667 (-0.20) | 0.668 (-0.21) | 0.670 (-0.18) | 0.670 (-0.23) | 0.510 (-0.46) |
| J03WOY | 0.662 (-0.16) | 0.696 (-0.14) | 0.691 (-0.12) | 0.695 (-0.16) | 0.450 (-0.47) |
| J03WPY | 0.677 (-0.20) | 0.684 (-0.18) | 0.689 (-0.16) | 0.691 (-0.18) | 0.503 (-0.42) |
| J03WQQ | 0.688 (-0.05) | 0.685 (-0.07) | 0.677 (-0.09) | 0.698 (-0.06) | 0.530 (-0.40) |
| J03WR9 | 0.598 (-0.21) | 0.604 (-0.19) | 0.592 (-0.15) | 0.621 (-0.13) | 0.528 (-0.36) |

The original-view per-match table is in `results/report_tables.md`; per-match rows for every arm and
view are in `results/per_match_results.csv`.

## 5. Pre-registered decisions (PREREG §6, `results/decision.json`)

| rule | statistic | result |
|---|---|---|
| H1a — `exp2_aug` − `exp1_ref` on `target_like`, CI lower > 0 | +0.0087 [+0.0010, +0.0182], 5/7 wins | **passed** |
| H1b — same on `original`, mean ≥ −0.02 | +0.0011 [−0.0076, +0.0102], 3/7 | **passed** |
| **H1** (H1a and H1b) | | **supported** |
| H2 count gate — every match of `exp2_aug_hsmm` within ±30 % on `target_like` | max abs count error 0.341 (J03WMX) | **failed** |
| H2 non-inferiority — `hsmm` − `const` on `target_like`, CI lower > −0.01 | −0.0022 [−0.0064, +0.0014], 4/7 | **passed** |
| **H2** (both) | | **not supported** |
| H3 — `ens3` − `exp2_aug` on `target_like`, abs Δ > seed spread and CI excludes 0 | +0.007490 [+0.0014, +0.0138], 5/7; seed spread 0.007483 | **passed** (margin 7×10⁻⁶) |
| Adoption of an exp_2 BAR-ATH candidate (iff H1) | | **adopted** (`exp2_aug`) |

Report-only, original view: H2 −0.0004 [−0.0033, +0.0015]; H3 +0.0095 [+0.0021, +0.0164], seed
spread 0.0057.

**exp_0 quality gates.** For every decoded SetSeq arm on both views, gain ≥ 0.05, CI > 0 and the
macroF1 check pass, and the ±30 % count gate fails on J03WMX (0.33–0.37). Only the argmax diagnostics
pass all four gates, at a cost of 0.02–0.03 spellF1. rasya's arms also fail the macroF1 check on
`target_like`.

## 6. BAR-ATH (label-free aggregates only)

Sources: `results/target_plausibility.json` and `target_candidate.json`. These numbers are report-only
and were never used for a choice.

| pipeline | dead share | possessions / min | mean possession (s) | possessions |
|---|---|---|---|---|
| exp_1 shipped (`exp_1_setseq_bd.csv`) | 0.384 | 4.13 | 8.96 | 437 |
| **`exp2_aug` (adopted candidate)** | 0.408 | 3.57 | 9.96 | 378 |
| `exp2_aug_hsmm` | 0.409 | 3.69 | 9.60 | 391 |
| `exp2_aug_ens3` | 0.408 | 3.75 | 9.48 | 397 |
| published prior | 0.407 | – | – | – |

* The candidate uses decoder T 1.28, λ_team 1.0, λ_dead 4.0, γ 0, selected on the 7 `target_like`
  out-of-fold predictions. exp_1 used λ_dead 0.5.
* The file passes the official validator. SHA-256 is `a59128e6…0aed`, and an independent second
  fit gives a byte-identical file. No graded frame abstains, and the public ground truth was never
  opened.
* Graded ball coverage of 0.593 matches the team EDA, so the loader is aligned.
* **Reading.** The exp_2 models move the dead share onto the prior. They also predict 9–14 % fewer
  possessions than exp_1 on BAR-ATH (378–397 vs 437). Every proxy arm already under-segments (§4), so
  lower counts are more likely a step in the wrong direction than a correction [Medium: proxy truth
  runs at about 4.6 possessions/min, but BAR-ATH's true rate is unknown].

## 7. Limitations and threats to validity

* **Small margins, 7 matches.** H1a's CI lower bound is +0.001, and the two other seeds of the same
  recipe give +0.001 and +0.008 against `exp1_ref`. H3 passes by 7×10⁻⁶. Read these as "no harm and
  likely a small gain", not as a robust effect.
* **The stress view is partly in-distribution for the augmented arms.** The off-pitch rule and the
  resulting coverage (0.395 vs 0.394) are matched by design. Only the dead-ball hiding differs:
  `target_like` places it from speed, `target_aug` from labels. So H1a is not a clean
  out-of-distribution test.
* **`target_like` encodes an assumption.** It hides the ball on slow play, so a missing ball becomes
  a dead-ball cue (ball visible on 15 % of dead frames vs 60 % of alive frames). If BAR-ATH's gaps
  are unrelated to dead time, the view overstates how well a missing ball signals dead time.
  BAR-ATH's coverage (0.593) is also higher than the view's.
* **The seed spread covers training seeds only.** The augmentation RNG depends on match and round,
  not on the model seed, so all three seeds see identical augmented data.
* **Ensemble vs single model.** `exp2_aug_ens3` against `exp1_ref` (+0.016) mixes ensembling with
  augmentation. The clean ensembling effect is ens3 − `exp2_aug` (+0.007 / +0.009).
* **The count gate still fails.** Under-segmentation is a model problem, not a decoder problem: no
  decoder reaches the argmax counts without losing about 0.03 spellF1. The HSMM lowers mean
  |count error| (0.199 → 0.185) but not the worst match.
* **Not blind.** exp_1's results, label-free view calibrations and one decoder smoke test came
  before freezing (PREREG §0). The test only changed the HSMM's functional form, no grid value.
* **`exp2_aug_multi` was not run.** Metrica has no licence. SkillCorner's labels are not comparable
  with IDSSE's (`DATA_SOURCES.md`). So no real cross-source transfer view exists yet.
* **Runtime.** Fits took 51–115 s depending on machine load. The 116 fits stayed within the ~150 cap.

## 8. Next steps (not done here)

1. **Real transfer view.** Harmonise SkillCorner labels (possession group → team possession; dead
   time from phases of play), resample to 25 fps, and run leave-one-source-out training in both
   directions.
2. **Fix under-segmentation in the model.** For example, use a count- or boundary-aware training
   loss, or a shorter window or stride. More decoder variants are unlikely to help (§7).
3. **Ensemble as candidate.** It is the best proxy arm. Adopting it needs its own pre-registered
   rule, because here it was report-only.
