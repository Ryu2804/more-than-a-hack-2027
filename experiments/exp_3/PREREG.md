# exp_3 pre-registration — train on the competition's possession definition

Written 2026-11-08/09, before any exp_3 model fit and before any model was scored on `L_ctrl`. Its
SHA-256 is stored in `results/protocol.json` with every constant below, read from the code. Each
notebook re-checks both, plus the judge and the pinned fusion snapshot, and stops on any difference.

## 0. What was seen before freezing (this experiment is not blind)

1. **exp_1 / exp_2 results** (`exp_1/REPORT.md`, `exp_2/REPORT.md`). On the 7 degraded IDSSE matches
   scored on provider labels, `target_like` view: `exp1_ref` 0.648, `exp2_aug` 0.657,
   `exp2_aug_ens3` 0.664. γ = 0 was selected in every fold. The HSMM did not fix counts.
2. **The planning session's findings (2026-11-08)**, from cached `exp2_aug` predictions. Notebook 00
   re-derives them.
   - Team assignment is the bottleneck: 5 fps ceiling 0.973, oracle dead/alive 0.679, oracle team
     where both are alive 0.944.
   - Misses are short possessions: 86 % of missed true possessions (IoU 0.5) last < 5 s; possessions
     < 2 s are 30 % of all, with recall 0.28; for these, the mean p(true team) is 0.58 when the ball is
     visible.
   - Provider labels ≠ the competition's definition. `L_track(τ)` runs at 3.83 / 3.02 / 2.49
     possessions/min for τ = 1 / 2 / 3 s, against 4.58 for provider labels, and changes 0.5 / 1.7 /
     3.1 % of frames. Without retraining, `exp2_aug` scores 0.693 / 0.683 against `L_track(1 / 2 s)`.
   - Decoder tweaks stay within noise.
3. **CHALLENGE.md § "Possession rules"**: changes only on controlled possession; contested or loose
   balls stay with the last team; proxies are "keeps the ball for a few seconds" or "more than two
   consecutive passes". The same section also says two things that bear on this experiment:
   - *"The ground truth is derived from corrected tracking data … a frame is `D` where the corrected
     tracking marks the ball dead, and otherwise takes the team the tracking names as owning it."*
     IDSSE's `BallPossession` is also a tracking-provider ownership field, so provider labels may be
     closer to BAR-ATH's ground truth than any controlled-possession relabel. That makes this a
     threat to H1's relevance (§8), not a reason to change the design.
   - The public 5-minute window is described as holding "17 possessions" (3.4/min). This count is
     **not** used for any choice. τ_hold = 2.0 s and the event list were fixed by the planning
     prompt before exp_3 started. The ground-truth file was never opened.
4. **IDSSE event files**: schema inspected, then synced, before freezing. Only the sync residuals
   were computed in this session, never `L_ctrl` statistics or scores (§2.3).
   - However, the sync fails and `L_ctrl` falls back to `L_track(2 s)`. So the planning numbers for
     τ = 2 s (3.02/min, 1.7 % of frames changed, `exp2_aug` 0.683 without retraining) are
     de facto `L_ctrl` statistics seen before freezing.
5. **Timing**: one decoder-selection run on one cached `exp2_aug` fold, provider labels. No choice
   was made from it.

## 1. Hypotheses

- **H1 (primary).** SetSeq trained, temperature-fitted and decoder-selected on `L_ctrl` (`exp3_ctrl_ens3`)
  scores higher against `L_ctrl` on the `target_like` view than the provider-trained ensemble with its
  decoder re-selected on `L_ctrl` (`exp3_ref_ens3`).
  - H1 is expected to be **partly mechanical**: training and test labels now match.
  - The real open question is whether `L_ctrl` matches BAR-ATH's ground truth, and no proxy can test
    that (§8).
  - The definition is a documented design decision taken from CHALLENGE.md. It was never tuned on
    the public window.
- **S1 — label sensitivity (report only).** Every arm on both views against provider labels,
  `L_track(1, 2, 3 s)` and `L_ctrl`. For each label set, selection is re-run on that same label set.
- **S2 — coverage sensitivity (report only).** A proxy for broadcast fusion. Clean ball positions are
  re-revealed on about 12 % of the `target_like` hidden-ball frames, and the saved weights predict
  this view without any training. Question: does spellF1 rise?
- **S3 — touch features (report only, stretch).** Runs only if the primary arms are done and at least
  6 h remain before the 1.5-day budget ends; otherwise it is skipped and REPORT.md says so.

## 2. Labels (codes 0 H, 1 A, 2 D, 3 unlabelled; 25 fps; per half, never across halves)

### 2.1 Provider

IDSSE `BallStatus` / `BallPossession`, through the frozen judge loader. Inside the halves of the
7 matches, no frame is unlabelled.

### 2.2 `L_track(τ)`, τ ∈ {1, 2, 3} s

Process the team runs of each half left to right. A team run shorter than τ (in frames, < 25·τ) is
relabelled to the previous team when all of these hold:

- it follows a run of the other team in the *output*;
- the next original run is that team, a dead ball, or the end of the half.

### 2.3 `L_ctrl` (primary)

Process the team runs of each half left to right. A team-k run is a **controlled possession** when any
of these hold:

- **(a)** it starts the half, or the previous run is dead ball;
- **(b)** it lasts at least `τ_hold = 2.0 s` (≥ 50 frames);
- **(c)** team k has a controlled on-ball event at a frame in [start − 0.5 s, end + 0.5 s].

Otherwise it takes the team of the previous (already relabelled) run. D frames never change.

**Event list for (c)** (DFL `events_raw` schema, inspected; `exp3_labels.py`):

| element | team attribute | condition |
|---|---|---|
| `Play` (pass or cross, also when nested in a set piece) | `Team` | `Evaluation` ∈ {`successfullyCompleted`, `successful`} |
| `ShotAtGoal` (any nesting) | `Team` | none |
| `BallClaiming` | `Team` | `Type` ∈ {`InterceptedBall`, `BallClaimed`, `BallHeld`} (not `Block`) |
| `TacklingGame` | `WinnerTeam` | `WinnerResult` ∈ {`ballClaimed`, `ballControlRetained`, `dribbledAround`} |

Excluded: `OtherBallAction` (touches and clearances), unsuccessful passes, tackles won only by
`ballcontactSucceeded`, `layoff` or `fouled`, `BallDeflection`, and `Delete` (which carries no target).
Club id → home/away comes from the match information file (`Role` home / guest).

**Sync.**

1. **Kick-off sync**, as the prompt specifies:
   `frame = first_frame_of_half + round((EventTime − kick-off EventTime) × 25)`.
2. **Snap refinement** (label-free, tracking-only): move each event to the frame within [−125, +50]
   frames of its synced frame (same half) where its player (`Player`, or `Winner` for tackles) is
   closest to the ball. Drop the event if that distance never reaches 2.0 m or the player is unknown.
3. **Acceptance.** Events are used iff, pooled over the 7 matches, the shot residual against the
   provider's own `CalculatedFrame` has median |r| ≤ 5 frames **and** p90 |r| ≤ 12 frames.
   - The criterion and the snap constants were fixed after the kick-off-only residuals were seen and
     before the snapped residuals were computed.
   - The kick-off-only residuals: 159 shots, median |r| 30, p90 99 frames, with events logged late by
     −3 s … +10 s.

**Result (seen before freezing).** After snapping: 152 shots, median |r| = 2 frames, p90 |r| = 57.5
frames.

- The p90 criterion **fails**, so events are **not used**.
- `L_ctrl` therefore uses (a) + (b) only (`EVENTS_USED = False`), the prompt's documented fallback.
- With (a) + (b) only, `L_ctrl` is identical to `L_track(2 s)`. Notebook 00 asserts this on all
  7 matches.
- The event code, its tests and the sync report stay in the experiment.
- Change residuals (claims and tackles won → nearest provider change into that team) are also
  reported: kick-off sync median |r| 31, p90 93 frames; snapped median |r| 13, p90 75 frames.

**Label-only statistics** (possessions/min, frames changed, agreement between `L_ctrl` and
`L_track(τ)`) are reported in notebook 00 before any model is scored on `L_ctrl` (notebook 01).

**5 fps targets** are `exp1_tokens.step_targets(L, token grid)` for every label set. They feed
training (state and boundary heads), temperature fitting and the selection objective.

## 3. Views

- **`original`** and **`target_like`** (primary): exp_2's views, unchanged (exp_2 `PREREG.md` §2),
  applied to round 0 of the held-out match.
- **`coverage`** (S2 only): start from `target_like` round 0. In every maximal run of hidden-ball
  graded frames of length L (same half), reveal the **clean** ball on one contiguous block of
  `round(0.12 · L)` frames at a uniformly random offset.
  - RNG seed: `20261108 + match index + 8 000 000`.
  - No labels are used. Frames where the clean ball is missing stay hidden.
  - Revealed share and ball visibility are reported.

## 4. Arms

| arm | predictions | fits |
|---|---|---|
| `exp3_ref_ens3` | mean p(H,A,D) and p(boundary) of cached `exp2_aug`, `exp2_aug_s1`, `exp2_aug_s2` (`exp_2/cache/predictions`, fingerprints hash-checked) | 0 |
| `exp3_ctrl`, `_s1`, `_s2` | `exp2_aug` recipe (round 0 original + rounds 1–2 `target_aug`, mirror, `TrainConfig()`), seeds 20261108 / 09 / 10. Training `state` / `boundary` come from `L_ctrl`; inputs and token caches are unchanged (exp_1 / exp_2 caches) | 84 |
| `exp3_ctrl_ens3` | per-step mean of the three `exp3_ctrl` seeds | 0 |
| `exp1_ref`, `exp2_aug` (reference) | cached exp_2 predictions | 0 |
| `baseline`, `rasya_c06`, `rasya_c08`, `ryu_final` (reference) | no fitting; frozen constants (exp_1 / exp_2) | 0 |

- **Decoder:** exp_1 main for every SetSeq arm: `D.select_config` with `D.GAMMAS` = {0, 0.5, 1, 2} and
  λ_team, λ_dead ∈ {0.5, 1, 2, 4, 8, 16}.
- **Fingerprint:** the prediction-cache fingerprint of each `exp3_ctrl*` arm includes the arm, recipe,
  config, the label set name, `EVENTS_USED`, and the SHA-256 of `exp3_labels.py`, `exp2_views.py`,
  `exp1_net.py` and `exp1_tokens.py`.
- **Weights:** each fit saves its weights, and each fit predicts both views of its held-out matches.

## 5. Protocol (exp_2's, with the evaluation label set made explicit)

- **Nesting:** nested leave-one-match-out per view and per label set.
  - The 21 pair models give inner validation predictions; the 7 outer models give the held-out
    predictions.
  - Temperature (NLL on 5 fps targets) and decoder config are chosen on the six inner validation
    matches with exp_0's objective (mean spellF1 − 0.15 × mean excess |count error| above 0.30).
  - Both use the **evaluation label set** of that analysis: `L_ctrl` for the primary analysis, the
    respective label set in S1.
  - The held-out match is scored once with the official `possession_metrics.py`, against the same
    label set.
- **Statistics:** paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval.
- **S2:** each fold's outer model predicts the `coverage` view. It is decoded with that fold's config
  selected on the `target_like` inner validation (same label set), so only the input changes. This is
  what fusion does on BAR-ATH.

### 5.1 S3 design (only if run; report only)

**Features.** Four extra global features per 5 fps step at frame i, computed from the 25 fps frames
[i − 4, i] of the same half on the same observed (degraded / view / augmented) match:

| feature | definition |
|---|---|
| `home_min_dist` | minimum home player–ball distance (m) / 10, clipped to [0, 3]; 3 when the ball is never visible in the window |
| `away_min_dist` | same for away players |
| `peak_dv` | max of \|v_f − v_{f−1}\| (m/s) / 20, clipped to [0, 3], with v_f = (b_f − b_{f−1}) × 25 and the ball visible on all frames used; 0 if undefined |
| `peak_team` | team of the nearest player at that peak frame: +1 home, −1 away, 0 if undefined |

**Mirror.** The features are in the canonical home/away frame. The mirror swaps `home_min_dist` and
`away_min_dist`, negates `peak_team`, and leaves `peak_dv` unchanged. A unit test checks this on a
synthetic match against the mirrored match.

**Arm `exp3_touch`.**

- Seed 20261108, the `exp3_ctrl` recipe and `L_ctrl` targets, with the 4 columns appended to exp_1's
  global vector.
- 28 nested fits.
- exp_1's modules are not edited. The S3 process temporarily extends `exp1_net`'s global-feature
  constants (names, mirror sign and swap lists) for its own fits only.

**Report.** Δ against `exp3_ctrl` (seed 0) on both views, `L_ctrl` and provider labels, with the
paired CI.

## 6. Decision rules (fixed now)

- **H1:** Δ = spellF1(`exp3_ctrl_ens3`) − spellF1(`exp3_ref_ens3`), `target_like` view, `L_ctrl`
  labels. H1 holds iff the paired 95 % CI lower bound is > 0.
- **Report only, never a decision:**
  - the same Δ on `original`;
  - Δ against `exp2_aug` (re-selected on `L_ctrl`) on both views;
  - single seeds and seed spread;
  - every S1 label set;
  - S2 (`coverage` − `target_like`, per arm, paired);
  - exp_0 quality gates against the same view's `baseline` on `L_ctrl`, including the ±30 % count gate;
  - mean / max |count error|.
- **Adoption tree (BAR-ATH candidate).**
  - The candidate is `exp3_ctrl_ens3` if H1 holds, otherwise `exp3_ref_ens3`.
  - In both cases:
    - Decode with the final config selected on the 7 `target_like` out-of-fold predictions against
      `L_ctrl` (`D.select_config`, `D.GAMMAS`).
    - Use full-data models: for `exp3_ctrl_ens3`, 3 new fits on all 7 matches; for `exp3_ref_ens3`,
      exp_2's cached full-data models, fingerprint-checked.
    - Write `predictions/exp_3_candidate.csv` and pass the official validator.
    - Rebuild byte-identically from **independent fits** of the same 3 members (3 more fits).
    - Report aggregates only. **No public check** (`exp_1/results/public_check_started.json` exists).
- **Broadcast fusion (BAR-ATH input), fixed now:**
  - Run exp_0's `fusion_candidate`, pinned verbatim in `judge/exp0_fusion_candidate.py` (SHA-256 in
    `judge/fusion_pin.json`). It accepts the broadcast ball per half only on label-free geometric
    agreement: ≥ 1000 overlap frames, median ≤ 1.5 m, p90 ≤ 5 m, homography valid and reprojection
    error < 5.
  - **Replay guard:** keep a recovered frame t only if, with p and n the nearest tactical
    ball observations before and after t in the same half, both exist and
    |b(t) − ball(p)| ≤ 40 m/s · (t − p)/25 and |b(t) − ball(n)| ≤ 40 m/s · (n − t)/25.
  - The candidate uses the fused input iff `fusion_candidate` accepts **both** halves; otherwise it
    uses tactical only.
  - Report recovered frames before and after the guard, and the candidate's aggregates on both
    inputs.
- **S3** never changes the candidate.

## 7. BAR-ATH plausibility (label-free, report only, never a choice)

Report dead share, possessions per minute and mean possession duration for:

- exp_1's shipped file (`exp_1/results/target_summary.json`);
- exp_2's candidate (`exp_2/results/target_candidate.json`);
- both exp_3 ensembles, on the tactical and on the fused input.

Compare against the 0.407 dead-share prior and ryu's 2.97 possessions/min.

## 8. Threats stated in advance

- **`L_ctrl` vs BAR-ATH's ground truth.** CHALLENGE.md says the ground truth takes the owner named by
  corrected tracking. If BAR-ATH's provider names owners the way IDSSE's `BallPossession` does, the
  provider labels are the better proxy, and adopting an `L_ctrl`-oriented candidate would hurt. The
  adoption tree picks an `L_ctrl`-decoded candidate either way; S1 shows how much each choice costs
  under the other label definitions.
- **The fallback removes events.** `L_ctrl` = `L_track(2 s)` merges every uncontrolled run shorter than
  2 s, including real short possessions certified by a successful pass or interception.
- **Seven matches, one league, one provider.**

## 9. Budget and hygiene

- **Budget:** 84 proxy fits + 3 full-data fits + 3 rebuild fits = 90, plus 28 if S3 runs. Cap 150.
  When the candidate is `exp3_ref_ens3`, its full-data models are cached (0 fits), and its rebuild
  re-fits the 3 exp_2 members with exp_2's recipe (3 fits).
- **BAR-ATH:** read only in notebook 05's kernel. That kernel prints aggregates only; its executed
  notebook is never opened, and results are read from `results/*.json`.
- **Never:** public ground truth, hand labels, writing under `data/`, or a frame-level BAR-ATH export
  other than the gitignored candidate CSV.
