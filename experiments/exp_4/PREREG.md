# exp_4 pre-registration — SetSeq + touch features + auxiliary hidden-ball head

Written 2026-11-10, before any exp_4 model fit. Its SHA-256 is stored in `results/protocol.json`
together with every constant below, read from the code (`exp4_common.frozen_constants`). Each
notebook re-checks both, plus the frozen judge, and stops on any difference.

## 0. What was seen before freezing (this experiment is not blind)

1. **exp_1–exp_3 results** (`exp_1/REPORT.md`, `exp_2/REPORT.md`, `exp_3/REPORT.md`, exp_3 `PREREG.md`).
   Mean spellF1 on 7 held-out degraded IDSSE matches, `target_like` view, decoder re-selected per
   label set: `exp3_ref_ens3` 0.664 (provider) / 0.748 (`L_ctrl`); `exp2_aug` 0.657 / 0.735;
   `exp1_ref` 0.648 / 0.729; `exp3_ctrl_ens3` 0.594 / 0.742.
2. **exp_3 diagnostics.** Team assignment is the bottleneck (oracle team where both alive 0.927 vs
   oracle dead/alive 0.677); 87 % of missed possessions last < 5 s; possessions < 2 s are 30 % of all
   with recall 0.28; 42 % of misses have the ball visible on ≥ 80 % of frames, where the true team gets
   p ≈ 0.60. Clean ball re-revealed on 12 % of hidden frames: +0.007 to +0.017 (S2).
3. **exp_3 S3 (one seed, `L_ctrl` training).** Touch features: +0.012 [−0.004, +0.027] on `L_ctrl` and
   +0.013 [−0.001, +0.030] on provider labels (`target_like`) against `exp3_ctrl` seed 0. Seed spread of
   that recipe ≈ 0.010; 3-seed ensembling adds +0.014 to +0.018.
4. **Before freezing, in this session (no exp_4 model scored):**
   - Unit tests on synthetic arrays. The bit-exact test found a bug (seeding the aux head with
     `torch.manual_seed` also reseeded the CUDA generator, which drives the GRU dropout masks); it was
     fixed so that only the CPU generator is forked. All tests then passed (16 after item 5).
   - One timing / code-path check: `exp3_ref_ens3` re-evaluated on provider labels, `target_like`, with
     exp_3's code reproduced exp_3's stored per-match rows and final config exactly (76 s).
   - Import checks: every exp_3 touch-feature cache and every reused exp_3 result file exists.
5. **Re-freeze before any fit (deviation, documented).** A first freeze (2026-11-10, kept as
   `results/protocol_superseded_1.json`) was followed by notebook 00's aux-target integrity check, which
   failed on one array: in J03WN1 round 2, half 1, the label-free flip sign of the *degraded* match differs
   from the clean match's (degradation moves the half's mean home x across halfway), so the observed tokens
   of that half are rotated and an aux target signed with the clean match would point the wrong way. The aux
   target now takes the flip sign of the observed match of the same corruption round (§3), a regression test
   covers it, and this file was re-frozen. No model had been fitted or scored; nothing else changed.

## 1. Hypotheses

- **H1 (primary).** SetSeq with the 4 touch features **and** a training-only auxiliary hidden-ball head
  (`exp4_full`, 3 seeds, provider labels; ensemble `exp4_full_ens3`) beats the exp_2 3-seed ensemble
  `exp3_ref_ens3` on provider labels, `target_like` view, and is non-inferior on `L_ctrl`.
- **Secondary (report only, never drive selection or adoption).**
  - Ablations `exp4_touch` (touch only) and `exp4_aux` (aux head only), 3 seeds each.
  - S1: every ensemble arm on `original` and `target_like` against provider, `L_track(1 s)`, `L_ctrl` and
    `L_track(3 s)` (`exp3_protocol.frame_labels`; `L_track(2 s)` ≡ `L_ctrl` under exp_3's fallback).
    Single seeds on provider and `L_ctrl`, both views.
  - Argmax diagnostic and recall (IoU 0.5) of true possessions < 2 s, per arm.
  - Generalisation gap per arm (§6).

## 2. Arms (fixed list; no post-hoc arms)

| arm | touch columns | aux head | seeds | fits |
|---|---|---|---|---|
| `exp4_full`, `_s1`, `_s2` → `exp4_full_ens3` | yes | yes (0.25 / 0.25) | 20261108 / 09 / 10 | 84 |
| `exp4_touch`, `_s1`, `_s2` → `exp4_touch_ens3` | yes | no | same | 84 |
| `exp4_aux`, `_s1`, `_s2` → `exp4_aux_ens3` | no | yes (0.25 / 0.25) | same | 84 |
| `exp3_ref_ens3` (reference) | – | – | exp_2's cached `exp2_aug`, `_s1`, `_s2` | 0 |
| `exp2_aug` (reference single seed) | – | – | exp_2 cached | 0 |

- **Ensemble:** per-step mean of p(H, A, D) and p(boundary) of the 3 seeds (`exp2_protocol.ensemble_predictions`).
- **Reference results** for `exp3_ref_ens3` and `exp2_aug` are exp_3's stored nested results
  (`exp_3/results/variant_*.json`, every S1 label set, both views), read in place. Their SHA-256 are
  frozen; notebook 00 re-computes one of them and requires equality.
- **Decoder for every arm:** exp_1 main, `D.select_config` with `D.GAMMAS` = {0, 0.5, 1, 2} and
  λ_team, λ_dead ∈ {0.5, 1, 2, 4, 8, 16}. Decoders are not tuned further.

## 3. Fixed design (nothing below is tuned)

- **Recipe.** `exp2_aug`: per training match, round 0 `original` + rounds 1–2 `target_aug` (cached tokens of
  exp_1 / exp_2), mirror on, `TrainConfig()` (window 300, batch 32, 14 epochs, coverage 2, lr 2e-3,
  weight decay 1e-4, width 64, hidden 64, boundary weight 0.5, pos weight 3), seed per arm. Training
  targets: provider state / boundary (the cached tokens' own targets; notebook 00 asserts they equal
  `exp3_protocol.step_labels(·, "provider")`).
- **Touch features** (exp_3 `exp3_touch`, by import, constants unchanged): per 5 fps step, from the
  25 fps frames [i − 4, i] of the same half on the observed match: per-team min player–ball distance / 10
  (cap 3), peak ball-velocity change / 20 (cap 3), team of the nearest player at that peak (±1, 0). Mirror:
  swap the distances, negate the team. Training and held-out caches are exp_3's, read in place.
- **Auxiliary head** (BallRadar-style; Kim et al., KDD 2023, arXiv 2306.08206). One linear layer from the
  BiGRU output (2 × 64) to 4 outputs, training only:
  - clean ball (x, y), normalised exactly as exp_1's `ball_x` / `ball_y` globals (centred, divided by the
    half-pitch, times the label-free per-half flip sign **of the observed match of that corruption round**,
    i.e. the sign the array's own tokens use); masked smooth-L1 (β = 1.0, PyTorch default),
    mean over the two coordinates and over steps where the clean ball exists; weight **0.25**;
  - team of the clean player nearest to the clean ball (0 home, 1 away); cross-entropy over steps where
    it exists; weight **0.25**.
  - Targets come from `load_clean(name)` of the **training matches only**, on every 5 fps step, including
    steps where the observed ball is hidden. Held-out matches never get aux targets.
  - Mirror: ball (x, y) negated, nearest team swapped, with the same per-window flip as the inputs.
  - Head init: seed `TrainConfig.seed + 9 000 000` on a forked CPU generator; the main parameters, the
    CPU and the CUDA RNG streams are exactly exp_1's.
- **Training loop.** `exp4_net.train` is `exp1_net.train` step for step (same RNG calls, `_batch`,
  `_mirror_batch`, `_loss`, AdamW, OneCycle, clipping) plus the aux term. With the aux weights at 0 and no
  touch columns it reproduces `exp1_net.train` bit-exactly (unit test, synthetic set).
- **Reproduction check (report only, 1 fit).** Notebook 01 trains exp_4's loop with aux off and no touch
  columns on exp2_aug's recipe and seed for one outer fold and compares it with exp_2's cached `exp2_aug`
  outer prediction. Bit-exact is expected; a max |Δp| above 1e-3 stops the run for diagnosis.

## 4. Protocol (exp_3's, unchanged)

- **Nesting.** Nested leave-one-match-out over the 7 IDSSE matches: per seed 21 pair models (inner
  validation) + 7 outer models (held-out). Temperature (NLL on 5 fps targets) and decoder are selected on
  the six inner validation matches with exp_0's objective (mean spellF1 − 0.15 × mean excess |count
  error| above 0.30), **on the evaluation label set of each analysis**. The held-out match is scored once
  with the official `possession_metrics.py`.
- **Views.** `original` and `target_like` (primary), exp_2's, applied to round 0 of the held-out match.
- **Statistics.** Paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval.
- **Evaluation never fits** (`fit=False`: a missing prediction is an error).

## 5. Decision rule and adoption tree (fixed now)

- **H1 passes iff both hold** (Δ = spellF1(`exp4_full_ens3`) − spellF1(`exp3_ref_ens3`), `target_like`,
  each label set with its own nested re-selection):
  1. provider labels: paired 95 % CI lower bound **> 0**;
  2. `L_ctrl`: paired 95 % CI lower bound **> −0.01** (non-inferiority).
- **Seed honesty.** Each single seed and the seed spread (max − min of single-seed means) are reported. If
  |Δ| on provider labels is smaller than the `exp4_full` seed spread, REPORT.md calls the result noise
  even if the CI passes (the decision itself still follows the rule above).
- **Ablations never adopt.** An ablation that beats `exp4_full_ens3` is reported, not adopted.
- **If H1 passes:** full-data `exp4_full` seeds 0–2 (3 fits on all 7 matches), BAR-ATH input from exp_3's
  `exp3_target` code by import (pinned fusion rule + replay guard; tactical input if fusion is rejected),
  aggregates only:
  - `predictions/exp_4_candidate_provider.csv`: decoder = final config selected on the 7 `target_like`
    out-of-fold predictions against provider labels;
  - `predictions/exp_4_candidate_ctrl.csv`: same, against `L_ctrl`;
  - both pass the official validator and rebuild byte-identically from 3 independent fits.
- **If H1 fails:** no exp_4 file is written; `results/decision.json` says so; exp_3's two files stay the
  team's options.
- **BAR-ATH plausibility** (dead share, possessions/min) is report-only. No public ground truth, no hand
  labels, no frame-specific constants.

## 6. Overfitting controls

1. This file is frozen before any fit (`freeze_prereg` refuses if any prediction exists).
2. Capacity and training are `TrainConfig()`; the only new numbers are the aux weights (0.25 / 0.25), the
   smooth-L1 β (library default) and the aux head seed offset. Touch constants are exp_3's.
3. Nested selection only; the BAR-ATH decoder is selected on the 7 out-of-fold predictions.
4. One primary comparison, fixed arm list.
5. **Generalisation gap (report only).** For every fit: final training NLL (raw p, T = 1, eval mode, no
   mirror, all training arrays, provider targets) and held-out NLL of its outer and pair predictions.
   gap = mean held-out NLL (original view; target_like also reported) − mean training NLL, per arm and per
   family (mean over 3 seeds). Reference: the exp_2 recipe (`exp2_aug`, `_s1`, `_s2`), training NLL
   recomputed from its saved weights (inference only). A family whose gap exceeds the reference gap by
   more than **25 %** is flagged in REPORT.md.
6. Seed honesty as in §5.
7. No BAR-ATH selection.

## 7. Budget and order

- 3 arms × 3 seeds × 28 = **252** proxy fits + 1 reproduction-check fit + 3 full-data + 3 rebuild fits
  (only if H1 passes) = 259. Cap **280**.
- Order: `exp4_full` (all 3 seeds) first, then `exp4_touch`, then `exp4_aux`. If time runs short,
  `exp4_aux` is dropped and REPORT.md says so. A second fit process may run in parallel; a per-job lock
  prevents a job from being fitted twice.
- Interpreter: the shared `.venv-torch`; no new venv, no installs. exp_1–exp_3 caches are read in place.

## 8. Hygiene

- BAR-ATH is read only in notebook 04's kernel and only if H1 passes; that kernel prints aggregates
  only; its executed notebook is never opened; results are read from `results/*.json`.
- Never: the public ground truth, hand labels, writing under `data/`, or a frame-level BAR-ATH export
  other than the gitignored candidate CSVs.
