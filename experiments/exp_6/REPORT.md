# exp_6 — recover the hidden ball from the tactical video (pre-registered)

Sources: `results/*.json` (aggregates only). BAR-ATH was read only inside the kernels of notebooks 00
(ATD + homography, no frame) and 02 (video + ATD); 04 was not run. Their executed notebooks were never opened;
no frame, crop or image of the match was displayed, described or written to disk. The public ground truth was
not opened (exp_1 spent the single check).

## 1. Verdict

**H1 fails. No exp_6 file is written; exp_3's two files stay the team's options.**

| clause | result | pass |
|---|---|---|
| G1 projection (ATD-visible graded balls inside the image) | 0.998 | yes |
| G2 detection quality, test blocks: precision@2 m ≥ 0.90 and recall@2 m ≥ 0.50 | **0.760 / 0.230** | **no** |
| G3 coverage of hidden graded frames ĉ ≥ 0.20 | 0.312 | yes |
| CI clause, measured setting, provider | Δ −0.125 [−0.138, −0.108], 0/7 | **no** |
| CI clause, measured setting, `L_ctrl` | Δ −0.145 [−0.162, −0.126], 0/7 | **no** |
| stress clause (mean Δ ≥ −0.005), provider / `L_ctrl` | −0.106 / −0.109 | **no** |

* **The detector is not good enough** [High; measured once on 18,952 test-block frames]. With τ at the 0.9
  fallback (no τ reached the 0.95 val precision the rule asks for), it agrees with the ATD ball within 2 m on
  76 % of accepted test frames and recovers 23 % of them.
* **At that quality the recovered ball would hurt, not help** [High on the proxy; Medium as a statement about
  BAR-ATH]. The measured simulation costs 0.125–0.145 spell-F1 on all 7 matches. The predicted possession count
  flips from −20 % (provider) to +17 % against the truth: wrong or jittery ball positions create spurious
  switches.
* **A clean ball would still pay** [High on the proxy]. Re-revealing the *clean* ball helps every match
  (provider +0.018 at 25 % of hidden frames, +0.035 at 50 %, +0.079 at 100 %). The value is there; this detector
  does not deliver it.
* Prior expectation (PROMPT): gates pass "about even" [Low]. They did not; G2 failed on both thresholds.

## 2. What was tested

**H1 (primary).** A ball recovered from the tactical video, at the quality measured on BAR-ATH, improves the
cached `exp3_ref_ens3` (exp_2's three `exp2_aug` seeds) on the IDSSE proxy: 7 held-out matches, saved outer-fold
weights, exp_3's stored per-fold `target_like` decoder selections, no fit and no re-selection.

* **Gates (BAR-ATH, label-free).** G1: ≥ 95 % of ATD-visible graded balls project into the image. G2: on the
  test blocks, precision@2 m ≥ 0.90 and recall@2 m ≥ 0.50. G3: accepted share of hidden graded frames ĉ ≥ 0.20.
* **CI clause.** At the measured setting S(ĉ, σ̂, f̂), the paired 95 % CI lower bound of Δ spell-F1 is > 0 on
  provider **and** on `L_ctrl`.
* **Non-harm clause.** At the stress setting (ĉ/2, 2σ̂, min(1, 2f̂ + 0.05)), mean Δ ≥ −0.005 on both.

**Pipeline.** 3-frame heatmap U-Net (16/32/64/128, 537k parameters) trained from scratch on automatic labels
(the ATD ball projected into the image), streamed from the video in 4 sequential decode passes → top-5 peaks per
frame → Viterbi linker with a 40 m/s speed limit → acceptance (score ≥ τ, on-path run ≥ 5 frames), τ picked on
val blocks → quality once on test blocks → proxy simulator S(c, σ, f) that re-reveals the clean IDSSE ball
inside `target_like` gaps in sub-runs drawn from BAR-ATH's accepted run lengths, with per-frame Gaussian noise
and false-positive runs that follow a player.

## 3. Protocol, deviations and what changed after freezing

* **Frozen** 2026-11-10T14:57:58Z (`PREREG.md` SHA-256 `dca69fd7…`) by notebook 00, before any detector
  training or simulated-view scoring. `results/protocol.json` holds the code-derived constants and the SHA-256
  of all 59 reused exp_2 / exp_3 / exp_5 files; every notebook re-verifies both.
* **Before freezing** (PREREG §0 item 6): G1 from ATD + homography, a 300-frame decode test (625 frames/s,
  byte-identical across two reads), a random-tensor GPU test and 17 synthetic tests. Two deviations were written
  into §0 first: the VOI grid moved from notebook 01 to 03 (it needs σ̂, f̂ and the run lengths that only 02
  produces), and the prompt's `exp3_views.coverage_view` is `exp3_coverage.coverage_view` in exp_3's code.
* **After freezing (code fixes, no design change; `verify_prereg` passes in every notebook).**
  1. `exp1_common.write_json` refuses NaN, and some report-only quality cells can be undefined (e.g. no accepted
     frame at a high τ). Results are written through `exp6_common.write_results`, which stores NaN as `null`;
     the simulator treats a missing σ̂ / f̂ as PREREG §0.6 item 8 says (not triggered: both were defined).
     `voi_grid` also drops duplicate cells if σ̂ or f̂ were exactly 0 (not triggered: 16 cells).
  2. nbclient does not shut down a kernel it was handed, so the kernels of 00 and 01 had stayed alive (the
     exp_5 hazard). They were stopped after checking their working directory; `run_notebook.py` now shuts its
     kernel down in a `finally` block (no kernel was left after 02 and 03).
  3. **02 ran three times; only the third finished.** The first launch was stopped by me about a minute in (to
     add fix 1; no checkpoint existed). The second was stopped by the app with its background task after
     ~2 min (no checkpoint). The third was launched detached (`setsid -f`) and ran to the end. All three started
     from the same frozen code and seed; the step-500 training loss was identical (0.26020) in the second and
     third runs.

## 4. Detector quality (BAR-ATH; ATD as the only reference)

Training: 55,967 train-block frames with a kept label, 167,901 samples per pass, 4 passes (10,496 steps),
2,432 s. Inference on 102,764 hidden + val + test frames, 6,812 s (66 ms per frame). Final weights
`state_sha256 e81f531d…`.

| blocks | frames | accepted | precision@2 m | recall@2 m | σ̂ (m, per axis) | f̂ |
|---|---|---|---|---|---|---|
| val (τ = 0.9) | 19,169 | 6,779 | 0.653 | 0.231 | 0.476 | 0.347 |
| **test (τ = 0.9)** | 18,952 | 5,742 | **0.760** | **0.230** | **0.483** | **0.240** |

**τ rule.** No τ in {0.1, …, 0.9} reached val precision 0.95, so τ = 0.9 (the pre-registered fallback; "the
gates decide").

| τ | val precision@2 m | val recall@2 m | val accepted | ĉ (hidden frames) |
|---|---|---|---|---|
| 0.1–0.5 | 0.573–0.574 | 0.538–0.540 | ~18,000 | 0.802–0.815 |
| 0.6 | 0.578 | 0.529 | 17,546 | 0.730 |
| 0.7 | 0.595 | 0.503 | 16,189 | 0.622 |
| 0.8 | 0.632 | 0.427 | 12,959 | 0.477 |
| 0.9 | 0.653 | 0.231 | 6,779 | **0.312** |

The linked track covers almost every frame at low τ, but only ~57 % of it is within 2 m of the ATD ball, and a
higher τ buys little precision. The score does not separate right from wrong tracks well.

**Coverage and runs (hidden graded frames).** ĉ = 0.312 (20,143 of 64,643 frames); the on-path share is 0.815
before τ. Ball visibility on graded frames would rise from 0.593 (ATD) to 0.720 if every accepted frame were
used. Accepted runs: 671; median 3 frames, mean 30.0, p10 1, p90 44, max 2,978; 86 % of accepted frames sit in
runs of ≥ 25 frames. A 2,978-frame (119 s) accepted run inside one gap suggests the track can lock onto a
static or slow target [Low; untested].

**Quality by distance from the ATD ball to the nearest ATD player (test blocks).**

| nearest player | frames | accepted | precision@2 m | recall@2 m |
|---|---|---|---|---|
| < 1 m | 5,220 | 1,684 | 0.918 | 0.296 |
| 1–2 m | 4,467 | 1,610 | 0.829 | 0.299 |
| 2–5 m | 5,643 | 1,371 | 0.610 | 0.148 |
| ≥ 5 m | 3,622 | 1,077 | 0.604 | 0.179 |

Agreement is best when the ball is at a player's feet and worst when it is away from players. That is where
passes, clearances and balls in the air happen; there, both the ATD and the detector project an airborne ball
onto the ground plane, and they can disagree by metres even when both "see" it [Medium; inference, not tested].

**Bridge consistency** (recovered position at a gap's first / last frame vs the ATD ball just outside the gap).

| edge | n | median | p90 | ≤ 2 m | ≤ 5 m |
|---|---|---|---|---|---|
| gap start | 32 | 0.85 m | 57.2 m | 0.59 | 0.69 |
| gap end | 22 | 2.75 m | 46.5 m | 0.50 | 0.55 |
| both | 54 | 1.19 m | 55.7 m | 0.56 | 0.63 |

Only 54 of ~300 gap edges have an accepted recovered position, and a third of those are > 5 m off: the track
often enters a gap on something other than the ball.

## 5. Simulator rechecks (IDSSE, notebook 01)

| check | result |
|---|---|
| S(c = 0) tokens = exp_2's cached `target_like` tokens (7 matches) | equal |
| S(c = 0) rows = exp_3's stored `exp3_ref_ens3` `target_like` rows (provider + `L_ctrl`) | equal (max abs diff 0); predictions bit-identical to exp_2's cache |
| exp_3's S2 coverage view regenerated in memory = exp_3's S2 rows | equal to 1.1e-16 (float parsing of the stored CSV) |
| S2 mean Δ, provider / `L_ctrl` (exp_6 / exp_3) | +0.0095 / +0.0095; +0.0069 / +0.0069 |
| `exp5_touch_ens3` `target_like` rows = exp_5's stored rows (context) | equal (max abs diff 0) |

The simulator's scoring path reproduces exp_3 exactly, so the Δ below come from the view change alone.

## 6. Primary decision (notebook 03, `results/decision.json`)

| setting | c | σ (m) | f | provider Δ [95 % CI], wins | `L_ctrl` Δ [95 % CI], wins |
|---|---|---|---|---|---|
| measured | 0.312 | 0.483 | 0.240 | −0.1246 [−0.1380, −0.1075], 0/7 | −0.1451 [−0.1619, −0.1259], 0/7 |
| stress | 0.156 | 0.966 | 0.529 | −0.1059 [−0.1187, −0.0922], 0/7 | −0.1092 [−0.1246, −0.0927], 0/7 |

* Base (`target_like`): mean spell-F1 0.664 provider, 0.748 `L_ctrl` (exp_3's numbers, reproduced).
* Measured setting: 0.540 provider, 0.603 `L_ctrl`. Visible-ball share rises from 0.395 to 0.584; 8,745
  sub-runs over the 7 matches, 25 % of revealed frames on false-positive runs. The count error goes from
  −20 % to +17 % (provider) and from −13 % to +24 % (`L_ctrl`).
* Failed: G2, both CI clauses, both stress clauses. Files: none.

## 7. VOI grid (report only; 16 cells, BAR-ATH run lengths)

| c | σ (m) | f | provider Δ [95 % CI], wins | `L_ctrl` Δ [95 % CI], wins |
|---|---|---|---|---|
| 0.12 | 0 | 0 | +0.0072 [+0.0018, +0.0122], 5/7 | +0.0108 [+0.0028, +0.0194], 5/7 |
| 0.12 | 0 | 0.240 | −0.0223 [−0.0289, −0.0154], 0/7 | −0.0221 [−0.0295, −0.0154], 0/7 |
| 0.12 | 0.483 | 0 | −0.0368 [−0.0447, −0.0258], 0/7 | −0.0414 [−0.0534, −0.0279], 0/7 |
| 0.12 | 0.483 | 0.240 | −0.0538 [−0.0617, −0.0446], 0/7 | −0.0596 [−0.0729, −0.0481], 0/7 |
| 0.25 | 0 | 0 | +0.0183 [+0.0118, +0.0226], 7/7 | +0.0186 [+0.0124, +0.0252], 7/7 |
| 0.25 | 0 | 0.240 | −0.0431 [−0.0535, −0.0329], 0/7 | −0.0422 [−0.0482, −0.0372], 0/7 |
| 0.25 | 0.483 | 0 | −0.0692 [−0.0781, −0.0604], 0/7 | −0.0905 [−0.1005, −0.0802], 0/7 |
| 0.25 | 0.483 | 0.240 | −0.0978 [−0.1089, −0.0858], 0/7 | −0.1138 [−0.1230, −0.1018], 0/7 |
| 0.5 | 0 | 0 | +0.0349 [+0.0231, +0.0426], 7/7 | +0.0334 [+0.0245, +0.0428], 7/7 |
| 0.5 | 0 | 0.240 | −0.0920 [−0.1102, −0.0741], 0/7 | −0.0915 [−0.1045, −0.0782], 0/7 |
| 0.5 | 0.483 | 0 | −0.1257 [−0.1376, −0.1138], 0/7 | −0.1606 [−0.1758, −0.1400], 0/7 |
| 0.5 | 0.483 | 0.240 | −0.1872 [−0.2058, −0.1671], 0/7 | −0.2069 [−0.2265, −0.1834], 0/7 |
| 1.0 | 0 | 0 | +0.0788 [+0.0592, +0.0943], 7/7 | +0.0562 [+0.0331, +0.0795], 7/7 |
| 1.0 | 0 | 0.240 | −0.1767 [−0.2011, −0.1483], 0/7 | −0.1750 [−0.1974, −0.1517], 0/7 |
| 1.0 | 0.483 | 0 | −0.1622 [−0.1765, −0.1481], 0/7 | −0.2055 [−0.2256, −0.1815], 0/7 |
| 1.0 | 0.483 | 0.240 | −0.2989 [−0.3234, −0.2724], 0/7 | −0.2857 [−0.3187, −0.2512], 0/7 |

* **A clean ball is worth up to +0.08 (provider) / +0.06 (`L_ctrl`)**, roughly linear in coverage. This matches
  exp_3 S2 (+0.010 at 12 %) and the prompt's prior (+0.02–0.05 at 40–60 %).
* **Either error alone erases it.** 24 % false-positive runs or 0.48 m of per-frame noise turns every cell
  negative, and the harm grows with coverage. A useful recovered ball therefore needs both high precision and
  low *jitter*, not just coverage.
* **Why 0.48 m of noise hurts this much** [Medium; mechanism inferred from the count error, not ablated]:
  the simulator adds independent noise per frame. The tokens' ball velocity is a central difference over ±2
  frames, so 0.48 m per axis becomes about 4 m/s of velocity noise per axis on a ball that should be still or
  rolling, and the "nearest player" flags flip. The models then predict spurious switches (count error turns
  positive). A real detector's error is probably correlated in time, so this noise model is likely pessimistic
  for σ. It was fixed in advance, and G2 fails independently of it.

## 8. `exp5_touch_ens3` context (report only)

The touch ensemble reproduces exp_5's stored `target_like` rows exactly through exp_6's path. At the measured
setting it loses more than the reference: provider −0.151 [−0.168, −0.137], `L_ctrl` −0.199 [−0.223, −0.176],
0/7 each. Touch features read ball acceleration, so they amplify position noise. This agrees with exp_5 §9's
warning that touch models react strongly to a different ball distribution on BAR-ATH.

## 9. BAR-ATH aggregates with the recovered ball

Not produced: H1 failed, so notebook 04 (fill, decode, files, rebuild) was not run, as PREREG §5 fixes. The
label-free numbers available from 02 are above: visibility 0.593 → 0.720 if every accepted frame were used,
ĉ 0.312.

## 10. Limitations

* **Gap frames are harder than test frames.** Quality is measured on test blocks, where the ATD itself saw the
  ball. The hidden frames are hidden because the ATD lost the ball there (occlusion, crowded boxes, balls in
  the air), so test precision / recall are an optimistic bound for the gaps, and ĉ is the only gap-side number.
  Bridge consistency at gap edges is the only (weak) check of gap-side positions.
* **Balls in the air project wrongly onto the ground plane.** Both the auto-labels and the recovered positions
  go through a ground-plane homography; an airborne ball's ground projection lands metres away from where the
  ball is above the pitch. The quality-by-distance table is consistent with this being a large part of the
  disagreement.
* **The ATD is itself noisy** (automatic, uncorrected). Precision and recall are agreement with another
  automatic system, not accuracy. Some "false positives" may be ATD errors, and the training labels carry the
  same errors.
* **The proxy gain is simulated.** The simulator draws independent per-frame Gaussian noise, false-positive runs
  that follow a random player and sub-run lengths from BAR-ATH. Real errors are correlated in time and
  concentrated on hard frames. It measures how the saved models react to more ball, not BAR-ATH's gain.
* **One fixed detector.** Architecture, loss, sampling and schedule were fixed in advance and trained once; the
  result says nothing about what a stronger detector could do, only that this one is not good enough.
* **Seven matches, one league, one provider**; paired match bootstrap, so the CIs are wide (they are far from
  zero here).
* **The label definition is still open** (provider vs `L_ctrl`); H1 required both, and both failed.
* **Post-hoc code review** (read-only, after the verdict): **no bug that could change a reported number.**
  It checked frame alignment, crop and heatmap axes, pixel ↔ metre broadcasting, the Viterbi, the
  quality and τ rules, the simulator's exact counts and noise units, the decision rule, resume determinism and
  hygiene. It flagged three effects that follow the spec literally rather than being bugs:
  1. Ground-plane labels and truth score a correct detection of a lofted ball as an error. This biases G2
     toward FAIL by an unquantified amount.
  2. Candidate → candidate transitions cost 0, and *none* emits at 0.5, so the path keeps any candidate scoring
     above 0.5. The ≥ 5-frame run rule is nearly inert, and τ is the only real precision control.
  3. Accepted runs are fragmented (median 3 frames), so the simulator reveals many short sub-runs with
     independent noise. This probably inflates the simulated damage. G2 fails regardless.

## 11. What this suggests (not done here; each needs its own pre-registration)

1. **A clean ball is worth +0.03–0.08 on the proxy, so ball recovery is still the biggest known lever**, but the
   bar is high: the VOI grid says false positives and jitter must be near zero before coverage pays.
2. **Detector-side:** the val precision plateau (~0.57 at any τ) points to label noise and airborne balls more
   than to the threshold. Options: a 3-D or height-aware label, training only on ATD frames whose ball is near
   a player, a learned ball/not-ball verifier on the linked track, or the broadcast camera (larger ball;
   PROMPT's exp_7 candidate).
3. **Model-side:** the saved SetSeq models were never trained on noisy recovered balls. Smoothing recovered
   tracks before tokenising, or training with simulated detector noise, would test how much of the harm is the
   models' sensitivity rather than the detector.

## 12. Reproduction and hygiene

* `python build_notebooks.py`, then `setsid --wait $PY run_notebook.py NB.ipynb > cache/logs/NB.log` for 00 → 03
  (04 only if `decision.json` says H1 passed). Interpreter: `.venv-torch`.
* Determinism: `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, seeded NumPy streams. Tests show
  bit-identical training and bit-identical resume from a checkpoint; the decode test shows byte-identical frames.
* Tests: 17 synthetic tests passed at freezing (`results/tests.json`). Two post-hoc checks were added at the
  end: the per-frame pixel → metre mapping of the top-5 candidates (the path behind the G2 numbers) and the
  simulator's noise being σ metres per axis (the path behind the measured-setting Δ). 19 pass.
* Write guard: the file-count + size/mtime digest of exp_0–exp_5 and both `data/` folders is unchanged after 02
  and 03 (`results/guard_02.json`, `guard_03.json`). No BAR-ATH frame, crop or image was written; per-frame
  arrays live only in `cache/` (gitignored).

## Files

* `PREREG.md` (frozen; §0 item 6 added before freezing), `REPORT.md`.
* Code: `exp6_common.py`, `exp6_data.py`, `exp6_detector.py`, `exp6_linker.py`, `exp6_recover.py`, `exp6_sim.py`,
  `exp6_decision.py`, `exp6_target.py`, `exp6_report.py`, `prefreeze_checks.py`, `build_notebooks.py`,
  `run_notebook.py`, `tests/test_exp6.py`.
* Notebooks: `00_protocol`, `01_simulator`, `02_detector`, `03_decision` (executed), `04_target` (not run).
  00 and 02 read BAR-ATH: do not open them executed.
* Results: `protocol.json`, `prefreeze_checks.json`, `tests.json`, `g1.json`, `inputs_summary.json`,
  `sim_rechecks.json`, `sim_rows_01.csv`, `detector_quality.json`, `decision.json`, `voi_grid.json`,
  `touch_context.json`, `sim_rows_03.csv`, `report_tables.md`, `write_guard_before.json`, `guard_02.json`,
  `guard_03.json`, `guard_final.json` (unchanged after everything).
