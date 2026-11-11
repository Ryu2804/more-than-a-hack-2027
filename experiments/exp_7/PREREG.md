# exp_7 pre-registration — a noise-robust possession model for the recovered ball

**Status: DRAFT.**

- Written 2026-11-11 at the end of the exp_6 session. No exp_7 code exists, no exp_7 model has been fitted
  and no exp_7 view has been scored.
- It becomes binding when `00_protocol.ipynb` freezes it: its SHA-256, the constants read from code and the
  SHA-256 of every reused file go to `results/protocol.json` (exp_3–exp_6 pattern). Every notebook re-checks
  them and stops on any difference.
- **Before freezing, the executing session may only:**
  - add items to §0, covering anything it sees or changes before freezing;
  - fill the code-derived constants into `protocol.json`.
- Any other edit to §1–§8 is a deviation and must be written into §0 before freezing.
- After freezing, nothing changes. A re-freeze is allowed only while no exp_7 fit exists, with the old
  protocol kept as `results/protocol_superseded_N.json`.

## 0. What was seen before freezing (this experiment is not blind)

1. **exp_0–exp_6 results.** Mean spell-F1, 7 held-out IDSSE matches, nested leave-one-match-out, `target_like`
   view unless stated.

   | exp | what | result |
   |---|---|---|
   | exp_1 | SetSeq (player-set encoder + BiGRU) | 0.656 (original view) |
   | exp_2 / exp_3 | `exp3_ref_ens3` (exp_2's 3 `exp2_aug` seeds) | 0.664 provider, 0.748 `L_ctrl` |
   | exp_4 / exp_5 | touch features | about +0.01 provider, the size of seed noise; BAR-ATH dead share shifted +0.08 |
   | exp_6 | ball recovered from the tactical video | **H1 failed**: G2 failed; the simulated recovered ball costs −0.125 provider / −0.145 `L_ctrl` |

2. **Evidence from exp_6 that motivates exp_7** (`exp_6/REPORT.md`).
   - **A clean ball is worth a lot.** Re-revealing the clean ball in `target_like` gaps, with saved weights and
     no retraining: provider +0.007 / +0.018 / +0.035 / +0.079 at 12 / 25 / 50 / 100 % of hidden frames;
     `L_ctrl` +0.011 / +0.019 / +0.033 / +0.056.
   - **The exp_6 detector, measured on BAR-ATH test blocks.** τ = 0.9 (fallback), precision@2 m 0.760,
     recall@2 m 0.230, σ̂ = 0.483 m per axis, f̂ = 0.240; ĉ = 0.312 of hidden graded frames; accepted runs
     median 3 frames, mean 30.
   - **The saved models are fragile to that ball.** At the measured setting `exp3_ref_ens3` loses 0.125
     (provider) and 0.145 (`L_ctrl`), 0/7 matches; stress setting −0.106 / −0.109. Either error alone
     (σ̂ = 0.48 m, or f̂ = 0.24) turns every VOI cell negative. The predicted possession count flips from −20 %
     to +17 %: the models read jitter and false positives as touches and switch possession.
     `exp5_touch_ens3` loses more (−0.151 / −0.199).
   - **The models were never trained on a recovered ball.** Their training views only remove the ball
     (`target_aug`), never add a noisy one.
   - **Caveats carried over.** The exp_6 simulator draws independent per-frame noise and fragmented sub-runs,
     so it probably overstates the harm. A post-hoc code review found no bug that could change the numbers.
   - **The recovered ball exists.** `exp_6/cache/recovered_ball.npz` holds accepted frames at τ = 0.9 from a
     deterministic detector (`state_sha256 e81f531d…`). exp_6 did not adopt it, because G2 failed.
3. **Known values.** ĉ, σ̂ and f̂ are known before exp_7 is designed, so exp_7 is not blind to its own
   evaluation setting. To limit fitting to them, the training-time error ranges in §3.2 are fixed, wide and
   generic, not centred on the measured values.
4. **In the drafting session:** no exp_7 code, no fit, no exp_7 view scored. The executing session appends
   here what it sees before freezing.
5. **In the executing session, before freezing (2026-11-11).** The "Status: DRAFT" line above is the drafting
   session's; this file is binding from the SHA-256 that `00_protocol.ipynb` writes to `results/protocol.json`.
   - **Done before freezing (allowed list only).** Code (`exp7_*.py`, `build_notebooks.py`, `run_notebook.py`,
     `prefreeze_checks.py`); 15 synthetic tests pass; import checks (exp_6 imports leave exp_7's environment and
     numba cache; exp_0–exp_6 and `data/` digests unchanged); token caches built (49 files, 835 MB: 7 matches ×
     2 simulated training rounds + 5 simulated evaluation views); a tiny synthetic fit (0.9 s). Seen: aggregate
     cache statistics only (`results/token_caches.json`: per-(match, round) draws, revealed / false-positive /
     flagged frames, ball visibility). **No exp_7 fit on IDSSE, no scoring of any view, no exp_7 prediction.**
   - **Checks passed before freezing.** The un-simulated `target_aug` view (rounds 1–2, 7 matches) rebuilt through
     exp_7's path equals exp_2's cached tokens and targets exactly (14 / 14). On every simulated view the flag
     equals the revealed frames exactly (no false-positive frame lacked its player).
   - **Implementation choices that make §3 concrete (not changes of design).**
     1. *Flag.* `ball_recovered` = 1 on graded frames where the view has a ball and its base (`target_aug`,
        `target_like` or the ATD) has none; on the 5 fps grid it is read at the step's frame.
     2. *Measured values.* Exact values from `exp_6/results/detector_quality.json` through `exp6_sim.measured` /
        `stress` (c 0.311604, σ 0.483201 m, f 0.239638; stress 0.155802 / 0.966403 / 0.529276), equal to exp_6's
        `decision.json`. §3.3's table shows them rounded.
     3. *Training targets.* Provider state / boundary from the training match's own labels via
        `exp1_tokens.step_targets` (exp_2's way; the check above shows they equal exp_2's cached targets).
     4. *Draw order.* One generator per (match, round), seed per §3.2: c, then σ, then f (uniform), then the
        same generator drives `exp6_sim.simulate`.
     5. *AR(1) view.* AR(1) noise (ρ = 0.9, stationary σ̂, restarted at each sub-run) replaces the per-frame noise
        on true-ball sub-runs only; false-positive runs keep exp_6's N(0, 1 m²) per frame. The generator is consumed
        exactly as in `exp6_sim.simulate`, so its sub-runs and false-positive runs equal the measured view's
        (test: ρ = 0 reproduces `simulate` bit for bit).
     6. *Clean views.* Clean 25 / 50 % = exp_6's VOI cells (c, 0, 0) with exp_6's seeds.
   - **Additions to §3.4 / §2 (report only; never in the decision, never adopted).**
     1. *Selection view for report-only views.* AR(1) selects on the measured view (a deployed decoder facing a
        different error model); clean 25 / 50 % select on their own view (a different detector would be
        re-tuned). Single seeds are evaluated with the `target_like` and measured selections only (seed spread).
     2. *`exp3_ref_ens3_msel`.* exp_2's saved pair + outer weights of the three `exp2_aug` members (no fit) on the
        measured, stress and AR(1) views, with nested selection on the measured view. It separates decoder
        re-selection from robustness training.
     3. *Reference rows.* `exp3_ref_ens3` with saved outer weights and exp_3's stored per-fold `target_like`
        selections (exp_6's path) on all six views. The `target_like`, measured, stress and clean rows must equal
        exp_6's stored rows (`sim_rows_03.csv`, tolerance 1e-12 for the CSV round trip) — a binding recheck in
        notebook 00, after freezing. AR(1) rows are report only.
     4. *Evaluation-path recheck (binding, notebook 02).* exp_7's nested evaluation applied to exp_2's cached
        `exp3_ref_ens3` `target_like` predictions must reproduce exp_3's stored rows, selections and final configs.
   - **Operational.** Evaluation runs sequentially in notebook 02, not alongside the fits (about 2–4 GB of RAM
     free on this machine). The token caches' SHA-256 are frozen with the protocol. The BAR-ATH rebuild (only if
     H1 passes) runs exp_6's `recover` in `exp_7/cache/rebuild`; it is resumable from its own checkpoints, which
     exp_6's tests show to be bit-identical.

## 1. Hypotheses

- **H1 (primary).** `exp7_robust_ens3` is a 3-seed SetSeq ensemble trained with simulated detector errors
  and a recovered-ball flag. Fed exp_6's recovered ball at its measured quality, it beats the current
  deployable pipeline, `exp3_ref_ens3` on `target_like` with no recovered ball, on the proxy. The win must hold
  on both provider and `L_ctrl`, and the new model must not lose when no ball is recovered.
- **Secondary (report only; never drives the decision or adoption):**
  - `exp7_robust_ens3` against `exp3_ref_ens3`, both on `target_like` (no recovery).
  - Clean-reveal cells (c ∈ {0.25, 0.5}, σ = f = 0): how much of the clean-ball value the robust model keeps.
  - **Transfer to a different error model** (simulator over-fitting check): measured c and f, but noise
    temporally correlated, AR(1) per axis with ρ = 0.9 and stationary σ̂. Scored for both ensembles.
  - Single-seed spread of `exp7_robust` (3 seeds).
  - BAR-ATH label-free aggregates with and without the recovered ball (dead share, possessions/min,
    mean p(dead)), if adopted.

## 2. Components (fixed list; no post-hoc variants)

| component | what | fits |
|---|---|---|
| A — `exp7_robust` × 3 seeds | exp_1 SetSeq, `TrainConfig()` except the seed; exp_2's `target_aug` recipe with simulated detector errors on the augmentation rounds (§3.2); one extra global input, `ball_recovered` (§3.1); provider labels | 84 proxy (3 × 28) + 3 full-data + 3 rebuild = 90 (cap 100) |
| S — exp_6 simulator | `exp6_sim.simulate` with exp_6's BAR-ATH run lengths (`exp_6/cache/run_lengths.npy`), read in place | 0 |
| reference | `exp3_ref_ens3`: exp_3's stored `target_like` rows (`selections`, `rows`) | 0 |
| recovered ball | `exp_6/cache/recovered_ball.npz`, read in place; regenerated from scratch by exp_6's code for the rebuild check | 0 (+1 detector rebuild if adopted) |
| decoders | exp_1 boundary-aware decoder; nested selection per label set (§3.4) | 0 |

## 3. Fixed design

### 3.1 Model

- exp_1 `SetSeq` with `TrainConfig()` unchanged (window 300, 14 epochs, coverage 2, lr 2e-3, width 64,
  hidden 64, mirror on); seeds 20261117, 20261118, 20261119.
- **One extra global feature, `ball_recovered`** ∈ {0, 1} per 5 fps step. It is 1 iff the ball at the step's
  frame comes from a recovery: a simulated reveal on the proxy, the detector fill on BAR-ATH. It is 0 on frames
  where the provider (proxy) or the ATD (BAR-ATH) supplies the ball, and 0 where there is no ball. The home /
  away mirror leaves it unchanged. It is appended to the globals the way exp_3 appended its touch features (a
  context manager around `exp1_net`'s global names); exp_1 is not edited.

### 3.2 Training views (provider labels; exp_2's recipe)

- **Round 0:** the original degraded view, flag 0, unchanged.
- **Rounds 1–2:** the `target_aug` view, then `exp6_sim.simulate` with parameters drawn once per
  (match, round) from a seeded generator:
  - c ~ U(0, 0.6), σ ~ U(0, 1.0) m, f ~ U(0, 0.5);
  - sub-run lengths from exp_6's BAR-ATH accepted run-length distribution; false-positive runs follow a
    clean-match player + N(0, 1 m²), as in exp_6;
  - `ball_recovered` = 1 on revealed frames.
- **Seeds:** parameter and simulation seed = 20261108 + 9000 + 1000 · round + match index (disjoint from
  exp_6's evaluation seeds, 20261108 + 7000 + match index).
- Labels and players are never changed; only ball observations and the flag.

### 3.3 Evaluation views (held-out matches; exp_6's seeds and run lengths)

| view | c | σ (m) | f | role |
|---|---|---|---|---|
| `target_like` | 0 | – | – | clause 2; reference rows |
| measured | 0.312 | 0.483 | 0.240 | clause 1 (H1) |
| stress | 0.156 | 0.966 | 0.529 | clause 3 |
| clean 25 % / 50 % | 0.25 / 0.5 | 0 | 0 | report only |
| AR(1) | 0.312 | 0.483 (stationary), ρ = 0.9 | 0.240 | report only (transfer) |

- The measured and stress views are exp_6's views exactly (same function, seeds and run lengths), with the flag
  set on revealed frames. The reference's rows on them already exist in `exp_6/results/sim_rows_03.csv`.

### 3.4 Decoders

- exp_1's grid (`LAMBDAS` × `LAMBDAS` × `GAMMAS`), temperature fitted per selection, nested per held-out match.
- **Selection view = deployment view.** For `target_like` scores, select on validation predictions of
  `target_like`. For measured and stress scores, select on validation predictions of the measured view: a
  deployed pipeline would pick its decoder for the recovered-ball input and then face worse quality.
- Final BAR-ATH configs: selected on the 7 out-of-fold measured-view predictions, per label set.

### 3.5 Ensemble

- Mean of the three seeds' p(H, A, D) and p(boundary), in seed order (exp_2 pattern).

## 4. Protocol

- 7 IDSSE matches; nested leave-one-match-out (exp_1's job keys: 21 pair + 7 outer fits per seed). Every fit
  predicts every evaluation view of its held-out matches with the same weights.
- Labels: `exp3_protocol.frame_labels` for provider and `L_ctrl`.
- Statistics: paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval.
- BAR-ATH: label-free only.

## 5. Decision rule and adoption tree (fixed now)

**H1 passes iff all three clauses hold on provider AND on `L_ctrl`:**
1. **Gain.** `exp7_robust_ens3` on the measured view minus `exp3_ref_ens3` on `target_like` (exp_3's stored
   rows): paired 95 % CI lower bound > 0.
2. **No harm without recovery.** `exp7_robust_ens3` on `target_like` minus `exp3_ref_ens3` on `target_like`:
   mean Δ ≥ −0.005.
3. **Stress.** `exp7_robust_ens3` on the stress view minus `exp3_ref_ens3` on `target_like`: mean Δ ≥ −0.005.

**If H1 passes:**
- **BAR-ATH input.** The tactical ATD (fusion off), with exp_6's recovered ball on accepted hidden graded
  frames only (quantised to 1 cm) and `ball_recovered` = 1 there. ATD-visible frames are never changed.
- **Model.** Three full-data `exp7_robust` fits (all 7 matches, exp_7's recipe).
- **Files.** `predictions/exp_7_candidate_ctrl.csv` and `predictions/exp_7_candidate_provider.csv`, each with
  its final measured-view config.
- **Plausibility guard (label-free; exp_5 lesson).** If a file's graded dead share lies outside
  0.407 ± 0.030 (the published prior), that file is not written and the reason is reported.
- **Each written file** passes the official validator and rebuilds byte-identically from scratch: three
  full-data refits **and** exp_6's detector retrained, inferred and linked in a fresh folder.

**If H1 fails:** no exp_7 file. `results/decision.json` names every failed clause, and exp_3's two files stay
the team's options.

**Never adopted:** any report-only view, any other decoder, any single seed.

## 6. Overfitting controls

1. **Freeze first.** No exp_7 fit and no exp_7 scoring before freezing.
2. **Fixed augmentation.** The ranges, run-length source, false-positive model and seeds are fixed in §3.2,
   wide and generic. They are never tuned after a result is seen.
3. **Disjoint seeds.** Training-time simulation seeds never equal exp_6's evaluation seeds.
4. **Nested decoders.** Selection never sees the held-out match.
5. **One primary comparison.** Every other view and arm is report-only.
6. **Not blind, stated.** σ̂, f̂ and ĉ were known while drafting (§0.3). The AR(1) view tests whether any
   gain is specific to the simulator's noise model.
7. **No BAR-ATH labels**, no public ground truth, no hand labels, no frame-specific constants.

## 7. Budget and order

- **Fits.** 84 proxy (about 57 s each, about 80 min), 3 full-data and 3 rebuild: 90, cap 100.
- **Token caches.** 7 matches × 2 augmented rounds for training (about 0.25 GB), plus 7 matches × 5 evaluation
  views (about 0.6 GB) in `exp_7/cache/`.
- **BAR-ATH, only if adopted.** 3 full-data fits (minutes), then the rebuild, which includes exp_6's detector
  (about 40 min of training + 2 h of inference).
- **Order.** `00_protocol` (freeze, tests, rechecks) → `01_fits` → `02_evaluation` → `03_decision` →
  `04_target` (only if H1 passes).
- **Limits.** Finish well before the 17 Nov deadline. Disk about 5.6 GB free; RAM about 19 GB; at most 3
  evaluation processes, never alongside fits if under 4 GB is free.

## 8. Hygiene

- **BAR-ATH access.** Only in the kernel of notebook 04 (and the detector rebuild it calls). Aggregates only;
  executed notebooks are never opened; results are read from `results/*.json`.
- **Never:**
  - display, plot, save or describe a frame, crop or image of the match;
  - print per-frame BAR-ATH arrays;
  - open `evaluation/gt/`, use hand labels, or write under `data/` or exp_0–exp_6;
  - export frame-level BAR-ATH output other than the gitignored candidate CSVs.
- **exp_6 is read in place** (`run_lengths.npy`, `recovered_ball.npz`, `exp6_sim`), SHA-256-checked, and added
  to the write guard.
