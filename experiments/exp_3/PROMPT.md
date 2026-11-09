# exp_3 session prompt — paste this into a new Claude Code session (cwd: barca/)

You are continuing the team's frame-level possession work (More than a Hack 2027, BAR-ATH). Build
**exp_3** in `exp_jeremy/exp_3/`. Deadline for the whole competition is **17 Nov 2026**, so exp_3 must
finish within about 1.5 days. Keep it focused: one primary hypothesis, nothing that cannot finish.

## Read first (code and docs only)

- `exp_jeremy/CLAUDE CODEX.md`: hard rules. **Data hygiene is absolute.** Never cat, print or open
  BAR-ATH data. Never read executed notebooks that touch BAR-ATH. Print aggregates only.
- `exp_jeremy/more-than-a-hack-2027/CHALLENGE.md` § "Possession rules": the label definition.
  Never open `evaluation/gt/`: the public check is spent (`exp_1/results/public_check_started.json`).
- `exp_jeremy/exp_2/REPORT.md`, `PREREG.md`, `DATA_SOURCES.md`, and the exp_2 modules
  (`exp2_*.py`). Reuse exp_1 and exp_2 by import: put `../exp_2` on `sys.path`, which puts `../exp_1`
  there too. **Never copy or edit them.**

## Where we stand

| arm (7 degraded IDSSE matches, nested leave-one-match-out) | original view | `target_like` view |
|---|---|---|
| exp_1 SetSeq (`exp1_ref`) | 0.656 | 0.648 |
| exp_2 `exp2_aug` (target-like augmentation; current adopted candidate) | 0.657 | 0.657 |
| exp_2 `exp2_aug_ens3` (3 seeds averaged; best arm, report-only) | 0.666 | 0.664 |
| ryu zero-shot / rasya C08 / baseline | 0.487 / 0.325 / 0.186 | 0.479 / 0.266 / 0.204 |

These are spellF1 scores against the provider's labels: IDSSE `BallPossession`, 4.58 possessions/min.

## What the planning session found (2026-11-08)

These numbers come from the cached exp_2 predictions of `exp2_aug`. The planning script is not part of
the repo, so re-derive the numbers in your notebook 00.

1. **Team assignment is the bottleneck, not dead ball and not frame rate.**
   - The 5 fps quantisation ceiling is 0.973.
   - With oracle dead/alive (true D, model's team), the score is 0.679.
   - With oracle team on the frames where both truth and prediction are alive (model's D), the score is 0.944.
2. **Misses are short possessions.**
   - 86% of missed true possessions (IoU 0.5) last under 5 s. Possessions under 2 s are 30% of all
     possessions, and their recall is 0.28.
   - 42% of the misses happen with the ball visible on at least 80% of the spell's frames.
   - For possessions under 2 s with the ball visible, the classifier gives the true team a mean
     probability of only 0.58. The decoder costs about 4 more recall points, so the frame model is the
     weak point.
3. **The proxy's labels are not the competition's labels.**
   - CHALLENGE.md says possession changes only on **controlled** possession. A touch without control,
     a deflection or a won aerial duel does not count, and contested or loose balls stay with the last
     team. Suggested proxies are holding the ball "a few seconds" or completing two or more
     consecutive passes.
   - IDSSE's 4.58 possessions/min includes touch-level spells.
   - A tracking-only relabel merges a team spell shorter than τ into the previous team, when that
     spell follows the other team and is followed by that team or by a dead ball. It gives 3.83/min at
     τ = 1 s, 3.02/min at τ = 2 s and 2.49/min at τ = 3 s, changing 0.5% / 1.7% / 3.1% of frames.
   - **Without retraining**, `exp2_aug` scores 0.693 (τ = 1 s) / 0.683 (τ = 2 s) against those labels,
     against 0.657 on provider labels. Count error goes from −20% to −5% / +20%.
   - So much of exp_1/exp_2's "under-segmentation" is a mismatch between the two label definitions.
     For reference, ryu's event-derived sources run at 2.97/min.
   - This analysis was seen before exp_3 was designed, so exp_3 is not blind. Say so in its PREREG.
4. Decoder tweaks stay below noise. γ = 0 was chosen in every fold of exp_1 and exp_2, and the HSMM did
   not fix counts. **Do not spend fits on decoders.**

## exp_3 hypothesis (one primary, pre-registered)

**H1: train on the competition's definition.** SetSeq trained, temperature-fitted and
decoder-selected on **controlled-possession labels** (`L_ctrl`, below) scores higher against `L_ctrl` on
the `target_like` view than the same recipe trained on provider labels whose decoder is re-selected on
`L_ctrl`.

- Comparison: `exp3_ctrl_ens3` vs `exp3_ref_ens3`.
- `exp3_ref_ens3` is the cached `exp2_aug_ens3` predictions, nested re-selection on `L_ctrl`, zero fits.
- Rule: paired 95% bootstrap CI lower bound > 0 (1000 draws, seed 20261108).

H1 is expected to be partly mechanical, because training and test labels now match. The real
uncertainty is whether `L_ctrl` matches BAR-ATH's ground truth, and no proxy can test that. State this
openly. The definition is a documented design decision taken from CHALLENGE.md, never tuned on the
public window.

**Secondary (report only, never drives selection)**

- **S1, label sensitivity.** Evaluate every arm against provider labels, `L_track(τ = 1, 2, 3 s)` and
  `L_ctrl`. Conclusions should not hinge on one label definition.
- **S2, coverage sensitivity.** Proxy for broadcast fusion. Re-reveal clean ball positions on 12% of the
  hidden-ball frames, as contiguous blocks inside gaps chosen without labels. Predict with the saved
  weights, with no training. Does spellF1 rise?
- **S3, touch features (time-boxed stretch).** Do this only if the primary arm has finished and at least
  6 h remain. Add sub-step ball-contact features from all 5 frames behind each 5 fps step:
  - per team, the minimum player–ball distance;
  - the peak change in ball velocity;
  - the team of the nearest player at that peak.
  Train one seed on `L_ctrl` and compare it with `exp3_ctrl` seed 0. The features must swap correctly
  under the home/away mirror; add a test for this. If you skip S3, say so in REPORT.md.

## `L_ctrl` (fix in PREREG before any fit)

Download the IDSSE event files `DFL_03_02_events_raw_*.xml` (7 files, about 0.6–0.8 MB each, CC BY 4.0).
They are on Figshare, DOI 10.6084/m9.figshare.28196177, and the API endpoint
`https://api.figshare.com/v2/articles/28196177/files` lists download URLs. Record them in
`exp_3/DATA_SOURCES.md`. None of floodlight, kloppy or databallpy is installed, so parse the XML with
lxml. Inspect the schema first and document the exact event types you use.

Start from the provider frame labels and split them into runs per half. A team-k run counts as a
**controlled possession** when any of these hold:

- (a) it starts the half or follows a dead-ball run. A restart always has a clear possessor
  (CHALLENGE.md).
- (b) it lasts at least `τ_hold` = 2.0 s. This reads CHALLENGE.md's "keeps the ball for a few seconds".
- (c) team k has a controlled on-ball event inside [start − 0.5 s, end + 0.5 s]: a successful
  pass or cross, a shot, a ball claim or interception, or a tackle won with the ball kept. Fix the list
  from the schema.

Otherwise, process runs left to right. If the run follows the other team's run, relabel it as that team.
This also covers "a touch that sends the ball out of play".

Sync events to frames from the half kick-off events (event time → `first_frame + round(s × 25)`). Report
the sync residual against provider possession changes. Time-box events to 3 h. If they cannot be synced,
fall back to (a) + (b) only (`L_track` with τ = 2 s) and say so.

Report label-only statistics before any model is scored on `L_ctrl`:

- possessions/min;
- frames changed;
- agreement between `L_ctrl` and `L_track(τ)`.

Never use the public window's count to choose τ or the event list.

## Design requirements

1. **Pre-register before any fit.**
   - `exp_3/PREREG.md` holds: what was seen before freezing (the four findings above), `L_ctrl`, arms,
     grids, decision rules, the adoption tree, and the fusion rule.
   - Store its SHA-256 and the code constants in `exp_3/results/protocol.json`, using exp_2's
     `freeze_prereg` / `verify_prereg` pattern. Every notebook verifies it.
2. **Same judge, views and protocol as exp_2.**
   - exp_1's frozen judge, round-0 held-out corruption, and exp_2's `target_like` view (primary) and
     `original` view.
   - 21 pair + 7 outer fits per seed, exp_0's selection objective, and the same bootstrap.
   - Decoder: exp_1 main (`D.select_config` with `D.GAMMAS`).
   - Selection objective and temperature use the **evaluation label set** of the analysis. For `L_ctrl`
     arms, that is `L_ctrl` at 5 fps.
3. **Arms**
   - `exp3_ref_ens3`: cached `exp2_aug` + `_s1` + `_s2` predictions from `exp_2/cache/predictions`,
     re-selected on `L_ctrl`. Zero fits.
   - `exp3_ctrl` seeds 20261108 / 09 / 10: the `exp2_aug` recipe (round 0 original + rounds 1–2
     `target_aug`, mirror, same `TrainConfig`) with training targets from `L_ctrl`. Inputs and token
     caches are unchanged. Only `state` / `boundary` come from `T.step_targets(L_ctrl, token_set)`.
     The prediction-cache fingerprint must include the `L_ctrl` code hash.
   - `exp3_ctrl_ens3`: per-step mean of the three seeds.
   - Also report single seeds, and the cached `exp1_ref` / `exp2_aug` re-selected on `L_ctrl`.
4. **Budget.** 84 fits for `exp3_ctrl` (3 × 28) + 3 full-data BAR-ATH fits + 1 rebuild, plus 28 if S3
   runs. Cap 150. Fits took 51–115 s depending on machine load. Run long notebooks in the background
   and wait for the completion notification. Check `nvidia-smi` first; idle exp_1 kernels may hold
   about 4 GB of GPU memory. Do not kill processes you did not start; ask instead.
5. **Adoption tree (pre-registered).** The BAR-ATH candidate is:
   - `exp3_ctrl_ens3` if H1 passes;
   - otherwise `exp3_ref_ens3` (the provider-trained ensemble, decoded with the config selected on
     `L_ctrl`).
   In both cases:
   - Use the final decoder config selected on the 7 `target_like` out-of-fold predictions against
     `L_ctrl`.
   - Write `exp_3/predictions/exp_3_candidate.csv`.
   - Run the official validator.
   - Rebuild byte-identically from independent fits.
   - Aggregates only. No public check.
6. **Broadcast ball fusion (BAR-ATH only, rule fixed in PREREG).**
   - Reuse exp_0's `experiment_target.fusion_candidate`. exp_0 is still being edited, so pin a verbatim
     snapshot of that function in `exp_3/judge/` with a SHA-256, as exp_1 did.
   - It accepts the broadcast ball only on label-free geometric agreement (median ≤ 1.5 m and
     p90 ≤ 5 m on overlap frames, homography reprojection error < 5).
   - Add a **replay guard**: keep a recovered broadcast frame only if the broadcast ball is within
     40 m/s × Δt of the nearest tactical ball observation on both sides.
   - Use fused input iff acceptance passes in both halves. Report recovered frames and both
     tactical-only and fused aggregates.
7. **BAR-ATH plausibility (report only).** Dead share, possessions/min and mean possession duration for
   exp_1, exp_2 and exp_3 candidates. Compare against the 0.407 dead-share prior and ryu's 2.97/min.
   Never select on these.
8. **Notebooks.**
   - Build with `exp_3/build_notebooks.py` (taking notebook names as arguments).
   - Run with a safe `run_notebook.py` that prints a one-line failure summary.
   - Suggested order: `00_labels_protocol`, `01_reference_on_ctrl` (zero fits), `02_fits`,
     `03_evaluation`, `04_comparison`, `05_target`, and optionally `06_touch`.
   - `exp_3/.gitignore` lists `cache/`, `predictions/`, `kernels/`, `*.error.txt` and the raw IDSSE
     event XML.
9. **Tests** in `exp_3/tests/` on synthetic arrays:
   - `L_ctrl` rules (a)–(c), the merge direction, never across halves, and D untouched;
   - event-to-frame sync;
   - the fusion replay guard;
   - S3 feature mirror consistency (if built).

## Interpreter, disk, housekeeping

- Interpreter: `/home/jeremy-mboe/Documents/Kuliah/Sem3/Machine_Learning/.venv-torch/bin/python`.
  **Do not create a venv.** Do not `pip install` either; parse with lxml.
- Disk has about 9.5 GB free. Read the exp_1 and exp_2 caches in place.
- Do not commit or push. The user pushes to branch `jeremy` themselves.
- A cost hook reports session spend. Keep turns lean while fits run.

## Done means

1. `exp_3/PREREG.md` hash was recorded before any new fit and is verified in each notebook.
2. `L_ctrl` is built and tested, with label statistics reported before any model score on it. Events are
   either used or the fallback is documented.
3. Notebooks `00`…`05` are saved in `exp_3/` and execute without errors. IDSSE-only notebooks may be
   read. BAR-ATH outputs are aggregates only, inspected through the results JSON.
4. `exp_3/REPORT.md` contains:
   - every arm on both views, against `L_ctrl` (primary) and against provider labels and `L_track(τ)`;
   - paired deltas with 95% CIs against `exp3_ref_ens3` and against `exp2_aug`;
   - each pre-registered rule marked passed or failed;
   - count gates, the broadcast-fusion report and BAR-ATH aggregates;
   - an explicit limitations section on the label-definition assumption.
5. The candidate CSV is validated and rebuilt byte-identically. Tests pass. Nothing under `data/` is
   touched, there is no frame-level BAR-ATH export besides the gitignored CSV, and there is no
   public-GT access.

## Out of scope (next steps for the team, not this session)

- Auxiliary hidden-ball supervision, in the style of BallRadar (Kim et al., KDD 2023): predict the clean
  ball position and carrier during gaps. Spells mostly in ball gaps cause about half the misses.
- SkillCorner leave-one-source-out with harmonised labels.
- Integrating the final pipeline into the team repo (`scripts/build_submission.py`, `TECHNICAL_DOC.md`).
