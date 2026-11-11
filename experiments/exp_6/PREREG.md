# exp_6 pre-registration — recover the hidden ball from the tactical video

**Status: DRAFT.**

- Written 2026-11-10 at the end of the exp_5 session, before any exp_6 code exists and before any BAR-ATH video frame has been decoded.
- It becomes binding when `00_protocol.ipynb` freezes it: its SHA-256 and the constants read from code go to `results/protocol.json` (the `freeze_prereg` / `verify_prereg` pattern of exp_3–exp_5).
- Every notebook re-checks both and stops on any difference.
- **Before freezing, the executing session may only:**
  - add items to §0, covering anything it sees or changes before freezing;
  - fill the code-derived constants into `protocol.json`.
- Any other edit to §1–§8 is a deviation and must be written into §0 before freezing.
- After freezing, nothing changes. A re-freeze is allowed only while no detector has been trained and no simulated view has been scored, with the old protocol kept as `results/protocol_superseded_N.json`.

## 0. What was seen before freezing (this experiment is not blind)

1. **exp_0–exp_5 results.** Mean spell-F1, 7 held-out IDSSE matches, nested leave-one-match-out, `target_like` view unless stated.

   | exp | what | result |
   |---|---|---|
   | exp_0 | LightGBM, 88 features + count prior | 0.422 (original view) |
   | exp_1 | SetSeq (player-set encoder + BiGRU) | 0.656 (original view) |
   | exp_2 | target-like augmentation; 3-seed ensemble | `exp2_aug` 0.657; `exp3_ref_ens3` 0.664 |
   | exp_3 | re-decoding on `L_ctrl` | `exp3_ref_ens3` 0.748 on `L_ctrl`; training on `L_ctrl` failed |
   | exp_4 | touch features + aux hidden-ball head | H1 failed (+0.007); touch-only 0.679, report-only |
   | exp_5 | touch-only, fresh seeds | H1 passed narrowly (+0.010 [+0.0009, +0.0199]) |

2. **Evidence that motivates exp_6.**
   - **Team assignment is the bottleneck, not dead/alive** (exp_3 §3.3). With the model's dead/alive and an oracle team on alive frames, spell-F1 is 0.927; with an oracle dead/alive and the model's team, 0.677; the model alone, 0.657.
   - **About half of the missed possessions sit in ball gaps** (exp_3 §8); 42 % of misses have the ball visible on ≥ 80 % of their frames.
   - **Revealing the ball helps every arm without retraining** (exp_3 S2). Clean ball re-revealed on 12.0 % of hidden frames (visibility 0.395 → 0.468), saved weights, fold decoder configs fixed from `target_like`:
     - `exp3_ref_ens3`: `L_ctrl` +0.007 [+0.000, +0.014], provider +0.010 [+0.002, +0.017];
     - every arm: +0.007 to +0.017.
   - **BAR-ATH is missing the ball on 41 % of graded frames** (64,643 frames), 99.7 % of them in gaps longer than 5 s (mean 17 s). DATA.md: the ball is lost to "occlusions, crowded boxes and balls in the air".
   - **CHALLENGE.md, "Scope":** "Improving the provided data, or replacing it with your own detection and tracking, is entirely legitimate as long as it is done by algorithms, not by hand." DATA.md: "A system built only on the ball track is as legitimate as one that reads the video."
   - **CHALLENGE.md, "What is already settled":** the ground truth is derived mechanically from *corrected tracking* — `D` where it marks the ball dead, otherwise the team it names as owning the ball. The ball is the input that ground truth is built on.
   - **exp_5 §9:** on BAR-ATH the touch ensemble's mean p(dead) is 0.474 against 0.398 for `exp3_ref_ens3`. Touch models are therefore not used for exp_6's adoption.
3. **Inputs (metadata only; no frame decoded).**
   - Tactical video: `data/tactical/BAR-ATH_tactical.mp4`, h264, 1920×1080, 25 fps, 199,319 frames. Video frame *i* = submission row *i* = ATD `frame_id − 1` (DATA.md: "There is no offset to discover").
   - Homography: one 3×3 matrix per frame mapping normalised pitch (0–1) to pixels, valid on every tactical frame, median reprojection error 0.24 px during play.
   - Scale: about 17 px per metre along the pitch, so the ball (0.22 m) is about 3–4 px wide.
   - Tooling in `.venv-torch`: `cv2` 4.13, `torch`, `ffmpeg`. There are no pretrained ball-detector weights on disk, and installs are not allowed, so the detector is self-written and trained from scratch.
4. **What exp_6 can and cannot show.**
   - Detection quality is measured against the automatic ATD ball, only on frames where ATD has a ball. Those are easier than the gap frames, where ATD lost it. Gap frames have no reference at all.
   - The possession gain is measured only on the IDSSE proxy, with a simulator calibrated to the measured detection quality.
   - So a pass means: "a recovered ball of the measured quality helps the proxy model, and does not hurt it under a pessimistic version of that quality". It cannot prove a BAR-ATH gain.
5. **In the drafting session:** no exp_6 code, no detector, no simulated view, no BAR-ATH frame decoded. The executing session appends here what it sees before freezing.
6. **Executing session, 2026-11-10, before freezing (additions only; §1–§8 unchanged).**
   - **Seen.** `PROMPT_exp6.md`, `CLAUDE CODEX.md`, `CHALLENGE.md` (settled, scope, may use), `DATA.md`, `RULES.md`, exp_3 `REPORT.md` §3.3, §4.3, §6–§8, exp_5 `REPORT.md` §9, exp_5 `PREREG.md` §0 item 5, and the exp_1–exp_3 functions exp_6 calls. No executed notebook, no `evaluation/gt/`; no frame or image of the match was displayed, described or saved.
   - **Code.** `exp6_common.py` (environment forcing first, reused-file registry, freeze/verify, write guard), `exp6_data.py` (ATD through `exp1_target.load_target_match`, homography through the starter's `load` / `check_alignment`, splits, auto-labels, sequential `cv2` reader), `exp6_detector.py`, `exp6_linker.py`, `exp6_recover.py`, `exp6_sim.py`, `exp6_decision.py`, `exp6_target.py`, `prefreeze_checks.py`, `build_notebooks.py`, `run_notebook.py`, `tests/test_exp6.py`.
   - **Pre-freeze checks** (`results/prefreeze_checks.json`, aggregates only):
     - **G1 = 0.998** (94,029 of 94,214 ATD-visible graded balls project inside the image with the 8 px margin): passes. Computed from ATD + homography only; no frame decoded.
     - **Splits** (label-free): train 91 runs / 56,093 frames (55,967 with a kept label), val 29 / 19,169 (19,169), test 29 / 18,952 (18,893), hidden 150 runs / 64,643 frames.
     - **Geometry.** Projected pitch bounding box: width 1,920 px (median; min 1,704), height 886 px (median; 526–1,078). Inference region (box + 32 px) median 920 × 1,920 px. 95.2 % of projected ATD players fall inside the image.
     - **Units.** `Match.ball` is in metres in both loaders: IDSSE on the match pitch (J03WMX 105 × 68 m; ball p99 x / length 1.02), BAR-ATH because `JD.load_target` scales the normalised ATD by 105.3 × 68.0 (95.8 % of visible graded balls lie inside the pitch rectangle; EDA: 4.2 % outside). The simulator and the BAR-ATH fill both write metres; pixel ↔ pitch goes through normalised coordinates (÷ or × 105.3, 68.0).
     - **Frame mapping.** `JD.load_target` asserts `frame_id − 1 = 0 … 199,318`; the starter's `check_alignment` confirms one homography row per video frame from 0; the video reports 199,319 frames, 1920 × 1080, 25 fps. The first training pass reads to the end of the file and asserts the decoded count.
     - **Decode test.** 300 frames read twice: 625 frames/s; the bytes of 3 probe frames are identical across the two reads.
     - **GPU (random tensors).** 537,473 parameters; training step 119 ms (batch 64 of 9 × 128 × 128, peak 1.9 GB); inference 63 ms per frame at 920 × 1,920 (fp32, peak 0.8 GB). Revised time estimate: training ≈ 4 × 10 min; inference ≈ 102,764 frames × 63 ms ≈ 1.8 h (§7 said 0.5–1 h); about 5 h of compute in total with the rebuild, inside the budget.
     - **Feasibility by import.** exp_2's fingerprints (`5cea8203869e`, `01776e3c0b77`, `fb137265b072`) match its cached folders. All 21 outer-fold `.pt`, the 3 full-data `.pt`, the 7 `target_like_r0` token caches, exp_3's two `exp3_ref_ens3` `target_like` result files (their `selections` cover all 7 matches), `coverage_rows.csv`, `target_candidate.json`, `target_hedge_provider.json`, exp_5's two `exp5_touch_ens3` result files and its 21 touch `.pt` exist. Their 59 SHA-256 are frozen in `results/protocol.json`.
     - **Tests.** 17 synthetic tests pass. The file-count + size/mtime digest of exp_0–exp_5 and both `data/` folders was identical before and after every pre-freeze run.
   - **Deviations (recorded before freezing).**
     1. **The VOI grid moves from notebook 01 to notebook 03.** Every grid cell needs σ̂, f̂ and the BAR-ATH accepted run-length distribution, which exist only after notebook 02. Notebook 01 keeps the two rechecks. The grid stays report-only (§7 "Order").
     2. The prompt's `exp3_views.coverage_view` is `exp3_coverage.coverage_view` in exp_3's code (with `exp3_coverage.reveal_seed`); used unchanged.
   - **Clarifications (implementation choices fixed now; no rule change).**
     1. **Linker passes.** Train-block frames are not inferred and act as forced *none*. This equals one Viterbi pass per run of consecutive inferred frames within a half, which is how it is implemented.
     2. **Crops.** Positive: the ball's integer pixel sits at an offset drawn uniformly from {16 … 111} px, i.e. inside the central 96 × 96. Near-player negative: drawn uniformly among projected team players inside the image and ≥ 40 px from the projected ball ("rejected" = excluded); a frame with none gets no near-player crop. Every crop's target is the Gaussian at the projected ATD ball, so a negative crop that happens to contain the ball is labelled correctly. Crop areas outside the image are zero. Frames *t ± 1* are input context only (they may be hidden frames), never labels.
     3. **Shuffle buffer.** Kept on the GPU as uint8 crops; when full, a random slot is evicted into the current batch. At the end of each pass the buffer is drained in random order and the last partial batch is trained, so every pass sees the same number of samples and a checkpoint holds no buffer state. The cosine decay runs per step over the 4 passes. Resuming from a checkpoint is bit-identical (test).
     4. **Inference.** Per-frame pitch bounding box (projected pitch corners) + 32 px, clipped to the image, zero-padded to a multiple of 8. Logits outside the region are masked before the 7 × 7 NMS; candidates are ranked by logit; score = sigmoid(logit).
     5. **False-positive runs** follow a player of the **clean** IDSSE match (identity-consistent), drawn uniformly among players present on every frame of the sub-run.
     6. **Sub-run placement.** A drawn length longer than every remaining free stretch is truncated to the longest stretch (needed at c = 1.0). The last sub-run is truncated so that exactly round(c · |hidden|) frames are revealed. Sub-runs may touch.
     7. **Simulator seed.** One generator per (match, setting), seeded 20261108 + 7000 + match index.
     8. **No accepted test frame.** σ̂ counts as 0 and f̂ as 1 in the simulator (G2 fails anyway).
     9. **Hidden graded frames** include 110,808–113,741 (the published half-2 block before kick-off, all `D`). They count in ĉ like other hidden graded frames; their codes are forced to `D` anyway.
     10. **Binding recheck.** If S(c = 0) does not reproduce exp_2's cached `target_like` tokens and exp_3's stored rows on both label sets, notebook 01 stops and nothing downstream runs. The S2 recheck is report-only.
     11. **`exp5_touch_ens3` context.** exp_4 / exp_5 modules are never imported (`exp4_common` / `exp5_common` redirect cache directories). exp_5's saved outer weights are loaded into exp_1's `SetSeq` built inside `exp3_touch.extended_globals()`; the aux head, unused at prediction, is dropped. Touch features come from `exp3_touch.touch_features` on each view. A recheck against exp_5's stored `target_like` rows is reported.
     12. **"Without the recovered ball"** (notebook 04): the same code on the plain tactical input must reproduce exp_3's two stored files byte for byte; reported.
     13. **Write guard.** exp_0–exp_5, `more-than-a-hack-2027/data` (BAR-ATH) and `exp_jeremy/more-than-a-hack-2027/data` (the metric repository's). Re-checked in notebooks 02, 03 and 04.
   - **The "Status: DRAFT" header is left as written** (only §0 additions are allowed). This file is binding from the freeze time recorded in `results/protocol.json`.
   - **Resources at freezing.** About 5.7 GB of 19.7 GB RAM available, 5.6 GB disk free, RTX 4060 idle.

## 1. Hypotheses

- **H1 (primary).** Ball positions recovered from the tactical video, at the quality measured on BAR-ATH, improve the cached `exp3_ref_ens3` on the proxy.
  - Comparison: the simulated-recovery view (§3.8) against the `target_like` view.
  - Measured with the same saved weights and the same per-fold decoder configs.
  - Both decision label sets, provider and `L_ctrl`.
- **Secondary (report only; never drives the decision or adoption):**
  - **Value-of-information curve.** Proxy Δ over a fixed grid of recovery coverage, noise and false-positive rate (§3.9).
  - `exp5_touch_ens3` on the simulated-recovery view, as context only.
  - **BAR-ATH label-free aggregates** with and without the recovered ball: ball visibility, dead share, possessions per minute, mean p(dead).
  - **Bridge consistency.** Distance between recovered positions near a gap's edges and the ATD ball just outside the gap.
  - Detection quality split by distance to the nearest player.

## 2. Components (fixed list; no post-hoc variants)

| component | what | fits |
|---|---|---|
| V — detector | 3-frame heatmap U-Net, trained on ATD-visible frames of the train blocks (§3.1–§3.4) | 1, plus 1 rebuild if adopted |
| L — linker | Viterbi over per-frame candidates with a speed limit (§3.5) | 0 |
| S — proxy simulator | re-reveals hidden IDSSE ball frames at measured coverage, noise and false-positive rate (§3.8) | 0 |
| possession models | `exp3_ref_ens3` members (`exp2_aug`, `_s1`, `_s2`): exp_2's saved outer-fold weights for the proxy and saved full-data weights for BAR-ATH, read in place | 0 |
| decoders | proxy: exp_3's stored per-fold `target_like` selections for `exp3_ref_ens3`, per label set. BAR-ATH: exp_3's final configs (`L_ctrl` from `target_candidate.json`, provider from `target_hedge_provider.json`) | 0 |

- No SetSeq model is trained in exp_6.
- No decoder is re-selected.

## 3. Fixed design (nothing below is tuned except τ, by the rule in §3.6)

### 3.1 Splits on BAR-ATH (label-free)

- Graded frames where ATD has a ball are grouped into maximal visible runs.
- Runs are numbered in time order within each half. Run *k* goes to **train** if *k* mod 5 ∈ {0, 1, 2}, **val** if *k* mod 5 = 3, **test** if *k* mod 5 = 4.
- Hidden frames (no ATD ball) are never used in training, not even as negatives, because they may contain the ball.

### 3.2 Auto-labels

- The ATD ball (normalised x, y) is projected to pixels with that frame's homography.
- A label is kept if it falls inside the image with an 8 px margin.
- The ATD ball's third channel is not used.

### 3.3 Detector

- **Input.** Frames *t−1, t, t+1*, RGB, scaled to [0, 1], stacked to 9 channels.
- **Network.** U-Net with 4 levels (16 / 32 / 64 / 128 channels), 3×3 conv + BatchNorm + ReLU, max-pool down, nearest-neighbour upsampling + conv up (all deterministic ops), one output logit per pixel.
- **Target.** A Gaussian with σ = 2.5 px at the projected ATD ball.
- **Loss.** BCE with `pos_weight` 50.
- **Training samples.** For each train-block frame with a label:
  - one positive 128×128 crop, with the ball at a uniform offset inside the central 96×96;
  - one random negative crop inside the pitch's image bounding box;
  - one near-player negative crop, centred on a randomly chosen projected player and rejected if it falls within 40 px of the projected ball.
- **Schedule.** 4 sequential passes over the video; shuffle buffer of 4,096 samples; batch 64; Adam at lr 1e-3 with cosine decay to 1e-5; seed 20261108.
- **Determinism.** `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Storage.** Frames are streamed from the video. No frame, crop or image is written to disk.

### 3.4 Inference

- Runs on every graded frame that is ATD-hidden or lies in a val or test block.
- Input is the full frame cropped to the pitch's image bounding box plus a 32 px margin.
- Output: the top 5 peaks after 7×7 max-pool non-maximum suppression, each with score = sigmoid(logit) and its pitch position (inverse homography, metres on 105.3 × 68.0).

### 3.5 Linker

- One Viterbi pass per half.
- **States per frame:** the 5 candidates plus *none*.
- **Emission cost:** −log(score) for a candidate; −log(0.5) for *none*.
- **Transitions:**
  - candidate → candidate costs 0 if the implied speed is ≤ 40 m/s (1.6 m per frame), and is forbidden otherwise;
  - candidate ↔ *none* costs 2.0;
  - *none* → *none* costs 0.
- **Acceptance.** A frame is accepted if its on-path candidate has score ≥ τ and belongs to an on-path candidate run of ≥ 5 frames.

### 3.6 Threshold τ (the only selected quantity)

- **Val blocks.** Inference and linking run on the val blocks as if their ATD ball were hidden.
- **Rule.** τ is the smallest value in {0.1, 0.2, …, 0.9} whose precision@2 m is ≥ 0.95. If none reaches it, τ = 0.9, and the gates decide.

### 3.7 Quality, measured once on the test blocks and on the BAR-ATH hidden frames

Test blocks, run as if hidden:
- **precision@2 m:** accepted frames within 2 m of the ATD ball, over all accepted frames.
- **recall@2 m:** accepted frames within 2 m, over all test-block frames.
- **σ̂:** per-axis RMS error, in metres, of the accepted inliers (≤ 2 m).
- **f̂** = 1 − precision@2 m.

Hidden graded frames:
- **ĉ:** the accepted share of hidden graded frames.
- **Run-length distribution:** the empirical distribution of accepted run lengths. It feeds the simulator.

### 3.8 Proxy simulator S(c, σ, f)

- **Base.** The `target_like` view of each IDSSE match: exp_2's `target_like` applied to the frozen round-0 degraded match.
- **Hidden frames.** Graded frames with no ball in the base but a ball in the clean match.
- **Reveal.**
  - Draw sub-run lengths from the BAR-ATH accepted run-length distribution.
  - Place them uniformly at random, without overlap, inside the hidden runs, until c × |hidden| frames are revealed (the last sub-run is truncated).
  - Position = clean ball + N(0, σ²) per axis, independent per frame.
- **False positives.** Each revealed sub-run independently becomes a false-positive run with probability *f*. Its positions are those of one randomly chosen visible player in that sub-run, + N(0, 1 m²) per frame.
- **Seed** = 20261108 + 7000 + the match's index.
- **Scoring.**
  - Tokens are built with `exp1_tokens.build_tokens`.
  - Each of the three `exp3_ref_ens3` members predicts with its saved `outer_<match>` weights; the predictions are averaged.
  - The result is decoded with that fold's stored `target_like` selection for the label set and scored once with the official metric.
- **Δ per match** = spell-F1(simulated) − spell-F1(`target_like`). The `target_like` rows are recomputed through the same code path; notebook 01 requires them to equal exp_3's stored rows.

### 3.9 Settings

| setting | coverage | noise | false-positive rate | role |
|---|---|---|---|---|
| measured | ĉ | σ̂ | f̂ | H1, CI clause |
| stress | ĉ / 2 | 2σ̂ | min(1, 2f̂ + 0.05) | H1, non-harm clause |
| VOI grid | {0.12, 0.25, 0.5, 1.0} | {0, σ̂} | {0, f̂} | report only (16 views) |

## 4. Protocol

- **Proxy.**
  - 7 IDSSE matches, nested-outer saved weights, fold configs fixed (exp_3 S2 protocol). No re-selection on any simulated view.
  - Labels: `exp3_protocol.frame_labels` for provider and `L_ctrl`.
- **Statistics.** Paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval.
- **BAR-ATH.** Label-free only. ATD, automatic and uncorrected, is the only reference for detection quality.

## 5. Decision rule and adoption tree (fixed now)

**Gates** (BAR-ATH, label-free):
- **G1 (projection sanity).** ≥ 95 % of ATD-visible graded balls project inside the image (§3.2).
- **G2 (detection quality, test blocks).** precision@2 m ≥ 0.90 and recall@2 m ≥ 0.50.
- **G3 (coverage).** ĉ ≥ 0.20.

**H1 passes iff all hold:**
1. G1, G2 and G3 pass;
2. at the measured setting, the paired 95 % CI lower bound of Δ is **> 0 on provider and > 0 on `L_ctrl`** (intersection-union test: both must pass);
3. at the stress setting, mean Δ ≥ −0.005 on both label sets (non-harm).

**If H1 passes:**
- **BAR-ATH input.** The tactical ATD, with the recovered ball filled in on accepted hidden frames only. ATD-visible frames are never changed, and broadcast fusion stays off (exp_3's pinned rule).
- **Possession model.** `exp3_ref_ens3`: exp_2's three cached full-data models.
- **Two files:**
  - `predictions/exp_6_candidate_ctrl.csv`, with exp_3's `L_ctrl` final config;
  - `predictions/exp_6_candidate_provider.csv`, with exp_3's provider hedge config.
- **Each file** must pass the official validator.
- **Rebuild.** The detector is retrained from scratch, then inference and linking re-run, and both files must come out byte-identical. Recovered positions are quantised to 1 cm before tokenisation.

**If H1 fails:** no exp_6 file is written. `results/decision.json` names every gate and clause that failed, and exp_3's two files stay the team's options.

**Never adopted:** touch models, any other decoder, any other τ, any VOI-grid setting.

## 6. Overfitting controls

1. **Freeze first.** No detector training and no frame decoding for training or scoring before freezing. G1 uses ATD and homography only, with no frames.
2. **Fixed detector.** Architecture, loss, sampling and schedule are fixed here. Only τ is chosen, on val blocks, by the rule in §3.6.
3. **Test blocks are used once**, for the quality numbers that parameterise the simulator.
4. **Possession models and decoders are unchanged.** No SetSeq fit; no decoder re-selection on any simulated view.
5. **Simulator parameters are measured, not tuned.** The stress setting is fixed now.
6. **No BAR-ATH labels.** No public ground truth, no hand labels, no frame-specific constants. ATD is input data, not ground truth.
7. **One primary comparison.** The VOI grid and the touch context are report-only.

## 7. Budget and order

- **Fits.** 1 detector, plus 1 rebuild if adopted. No possession-model fit.
- **Time (estimate).**
  - Decoding is about 10 min per sequential pass, so 4 training passes take about 1 h.
  - Inference on the hidden, val and test frames (about 100k frames) takes 0.5–1 h.
  - Each simulator evaluation takes 1–2 min: 7 matches × 3 members, inference only.
- **Order.** `00_protocol` (freeze, tests, G1) → `01_simulator` (recheck vs exp_3 S2, VOI grid) → `02_detector` (train, τ on val, quality on test, ĉ, G2/G3) → `03_decision` (measured and stress settings → `decision.json`) → `04_target` (only if H1 passes).
- **Limits.** Wall time ≤ 1.5 days, so it finishes well before the 17 Nov deadline. Disk is about 5.8 GB free: nothing but weights and per-frame arrays may be written. RAM is about 19 GB: at most 3 evaluation processes, and never alongside detector training if less than 4 GB is free.

## 8. Hygiene

- **BAR-ATH access.**
  - The BAR-ATH video and ATD are read only in the kernels of notebooks 02 and 04. Notebook 00 uses ATD and homography for G1, without frames.
  - Those kernels print aggregates only. Their executed notebooks are never opened.
  - Results are read from `results/*.json`.
- **Never:**
  - display, plot, save or describe a frame, crop or image of the match;
  - print per-frame BAR-ATH arrays;
  - write a frame, crop or image to disk.
- **Storage.** Detector weights and per-frame recovered-ball arrays stay in `exp_6/cache/` (gitignored).
- **Never:**
  - open `evaluation/gt/`;
  - use hand labels;
  - write under `data/` or exp_0–exp_5;
  - export frame-level BAR-ATH output other than the gitignored candidate CSVs.
