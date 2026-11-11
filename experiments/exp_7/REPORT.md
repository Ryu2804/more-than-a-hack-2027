# exp_7 — a noise-robust possession model for the recovered ball (pre-registered)

Sources: `results/*.json`, `results/ref_rows.csv`, `results/report_tables.md` (IDSSE aggregates only).
BAR-ATH was not read: notebook 04 did not run because H1 failed. exp_6's BAR-ATH-derived files were read in place
and SHA-256-checked; of them, only the aggregate run-length distribution entered a computation. The public
ground truth was not opened (exp_1 spent the single check).

## 1. Verdict

**H1 fails, on `L_ctrl` only. No exp_7 file is written; exp_3's two files stay the team's options.**

| clause (vs `exp3_ref_ens3` on `target_like`, exp_3's stored rows) | provider Δ [95 % CI], wins | pass | `L_ctrl` Δ [95 % CI], wins | pass |
|---|---|---|---|---|
| 1. gain: measured view, CI low > 0 | **+0.0137 [+0.0033, +0.0265]**, 6/7 | yes | +0.0022 [−0.0112, +0.0139], 4/7 | **no** |
| 2. no harm: `target_like`, mean ≥ −0.005 | −0.0007 [−0.0076, +0.0070], 3/7 | yes | **−0.0059** [−0.0171, +0.0038], 4/7 | **no** |
| 3. stress view, mean ≥ −0.005 | −0.0041 [−0.0090, +0.0000], 2/7 | yes | **−0.0054** [−0.0162, +0.0057], 3/7 | **no** |

`results/decision.json`: `h1_passed: false`, failed `gain:ctrl`, `no_harm:ctrl`, `stress:ctrl`; candidate files:
none.

* **Robustness training removes the harm.** [High on the proxy] On exp_6's measured recovered-ball view, the
  saved reference scores 0.540 provider and 0.603 `L_ctrl`. `exp7_robust_ens3` scores 0.678 and 0.751. That is
  +0.138 and +0.147 on the same input. The reference with its decoder re-selected on the measured view
  (`exp3_ref_ens3_msel`) reaches only 0.574 and 0.638. So decoder re-selection recovers about 0.03 of the loss,
  and the training recovers the remaining ~0.10.
* **Provider would pass; `L_ctrl` misses by small margins.** [Medium] No-harm misses by 0.0009 and stress by
  0.0004, both inside one seed's spread (≤ 0.0095 on `L_ctrl`). The `L_ctrl` gain is not distinguishable from 0.
  The rule is an intersection, fixed in advance, so H1 fails.
* **Prior** (PROMPT): H1 passes "about 1 in 4" [Low]. It failed, but much closer than the prior's reasoning
  expected. The prior assumed the robust model would struggle to remove the −0.125 loss. It removed all of it.

## 2. What was tested

**H1 (PREREG §1, §5).** `exp7_robust_ens3` (3 seeds; exp_1 SetSeq, `TrainConfig()` except the seed; one extra
global `ball_recovered`) is trained on exp_2's `target_aug` recipe. Rounds 1–2 add simulated detector errors:
c ~ U(0, 0.6), σ ~ U(0, 1) m, f ~ U(0, 0.5), drawn once per (match, round); exp_6's simulator; exp_6's BAR-ATH
run lengths. Fed exp_6's measured recovered ball, it must beat `exp3_ref_ens3` on `target_like` on provider
**and** `L_ctrl`, with no harm without recovery and under the stress setting.

**Views** (held-out matches, exp_6's seeds and run lengths): `target_like`; measured (c 0.3116, σ 0.4832 m,
f 0.2396); stress (0.1558, 0.9664, 0.5293); clean 25 / 50 %; AR(1) (measured c and f, ρ = 0.9 noise on true-ball
sub-runs, same sub-runs as measured). Selection view = deployment view (PREREG §3.4, §0 item 5): `target_like` →
`target_like`; measured, stress and AR(1) → measured; clean → its own view.

## 3. Protocol, rechecks and what happened after freezing

* **Frozen** 2026-11-11T01:29:38Z (`PREREG.md` SHA-256 `189874eb…`) by notebook 00, before any exp_7 fit or
  scoring. `results/protocol.json` holds the code-derived constants and the SHA-256 of 115 reused files (exp_1
  tokens, exp_2 weights and tokens, exp_3 rows, exp_6 caches / results / code) and of exp_7's 49 token caches.
  Every notebook re-verifies them.
* **Before freezing** (§0 item 5, added before the freeze): code, 15 synthetic tests, import checks, and the
  token caches. The 14 un-simulated `target_aug` views equal exp_2's cached tokens and targets exactly. On every
  simulated view the flag equals the revealed frames exactly. Additions to §2 / §3.4 were written into §0 first;
  all are report only (selection views for the report-only views, `exp3_ref_ens3_msel`, the two rechecks).
* **Binding recheck 1 (notebook 00).** `exp3_ref_ens3` with saved weights on exp_7's views, without the flag,
  through exp_6's scoring path. Its rows equal exp_6's stored rows on `target_like`, measured, stress and clean
  25 / 50 % (max abs diff 1.1e-16, from the CSV round trip). So exp_7's evaluation views are exp_6's.
* **Binding recheck 2 (notebook 02).** exp_7's nested evaluation on exp_2's cached `target_like` predictions
  reproduces exp_3's stored rows, selections and final configs exactly (provider and `L_ctrl`).
* **No code change after freezing.** No re-freeze. 00–03 each ran once, detached (`setsid -f`). The first
  watcher on 02 tracked the wrong PID; the run itself was not affected.

## 4. Fits and seed spread

84 proxy fits (3 seeds × 28 nested jobs), about 48 s each including the six-view predictions, 67 min in total; 0 full-data, 0 rebuild (H1
failed). Cap 100. Each job predicted all six views of its held-out matches with the same weights. The 22
nested evaluations took 35 min.

| view | label | seed 20261117 | seed 20261118 | seed 20261119 | spread | ensemble |
|---|---|---|---|---|---|---|
| `target_like` | provider | 0.6480 | 0.6486 | 0.6484 | 0.0006 | 0.6634 |
| `target_like` | `L_ctrl` | 0.7357 | 0.7351 | 0.7265 | 0.0092 | 0.7424 |
| measured | provider | 0.6657 | 0.6673 | 0.6635 | 0.0038 | 0.6779 |
| measured | `L_ctrl` | 0.7464 | 0.7519 | 0.7424 | 0.0095 | 0.7505 |
| stress | provider | 0.6465 | 0.6509 | 0.6426 | 0.0083 | 0.6600 |
| stress | `L_ctrl` | 0.7352 | 0.7354 | 0.7266 | 0.0088 | 0.7429 |

* **Ensembling is worth about +0.015** (provider) over any single seed. Each single seed is 0.013–0.022 below the
  reference on `target_like`. The clause margins (0.0004–0.0009 on `L_ctrl`) are an order of magnitude smaller
  than the seed spread. [High; 3 seeds only]

## 5. Report-only views (spell-F1; Δ vs `exp3_ref_ens3` on `target_like`)

| view | `exp7_robust_ens3` provider | `exp3_ref_ens3` provider | `exp7_robust_ens3` `L_ctrl` | `exp3_ref_ens3` `L_ctrl` |
|---|---|---|---|---|
| `target_like` | 0.6634 (−0.0007) | 0.6641 | 0.7424 (−0.0059) | 0.7483 |
| measured | 0.6779 (+0.0137) | 0.5396 (−0.1246) | 0.7505 (+0.0022) | 0.6032 (−0.1451) |
| stress | 0.6600 (−0.0041) | 0.5583 (−0.1059) | 0.7429 (−0.0054) | 0.6391 (−0.1092) |
| AR(1) | 0.6781 (+0.0140) | 0.5692 (−0.0949) | 0.7515 (+0.0032) | 0.6400 (−0.1083) |
| clean 25 % | 0.6879 (+0.0238) | 0.6824 (+0.0183) | 0.7652 (+0.0169) | 0.7669 (+0.0186) |
| clean 50 % | 0.7023 (+0.0382) | 0.6991 (+0.0349) | 0.7777 (+0.0294) | 0.7817 (+0.0334) |
| measured, reference re-selected on measured (`msel`) | – | 0.5740 (−0.0902) | – | 0.6381 (−0.1102) |

Reference rows: saved weights with exp_3's `target_like` selections (exp_6's path), except the `msel` row.

* **No simulator over-fitting detected.** [Medium] The AR(1) view (correlated noise, an error model the robust
  model never saw) scores the same as the measured view (+0.014 provider). The reference loses less on AR(1)
  (−0.095) than on measured (−0.125), which supports exp_6's guess that i.i.d. per-frame noise is the pessimistic
  case.
* **The robust model keeps the clean-ball value.** Relative to its own `target_like` score, clean 25 % adds
  +0.025 provider / +0.023 `L_ctrl` and clean 50 % adds +0.039 / +0.035. The reference gains +0.018 / +0.019 and
  +0.035 / +0.033 on the same views. The measured view adds +0.015 / +0.008: about half (provider) and a third
  (`L_ctrl`) of what a clean ball at the same coverage (~0.31, interpolated: +0.028 / +0.026) would add.
* **Stress costs a little.** Relative to its own `target_like`: −0.003 provider, ±0.000 `L_ctrl`.

## 6. Count errors (mean count-error ratio, predicted vs true possessions)

| view | robust provider | reference provider | robust `L_ctrl` | reference `L_ctrl` |
|---|---|---|---|---|
| `target_like` | −0.216 | −0.202 | −0.115 | −0.125 |
| measured | −0.210 | **+0.166** | −0.102 | **+0.235** |
| stress | −0.211 | +0.068 | −0.110 | +0.096 |
| AR(1) | −0.213 | +0.079 | −0.105 | +0.144 |

The spurious-switch mechanism exp_6 found is gone. With the recovered ball, the robust model's count error stays
at its `target_like` level. The saved models flip to over-segmentation. [High on the proxy]

## 7. BAR-ATH

Not produced: H1 failed, so notebook 04 (full-data fits, fill, plausibility guard, files, rebuild with exp_6's
detector retrained) was not run, as PREREG §5 fixes. No BAR-ATH aggregate exists for exp_7.

## 8. Limitations

* **Simulator over-fitting.** Training and evaluation share exp_6's simulator family: per-frame noise, false
  positives that follow a random player, BAR-ATH run lengths. AR(1) changes only the noise correlation. Real
  gap-side errors are concentrated on hard frames (occlusion, balls in the air) and may correlate with possession
  changes, which no view models. The gain is a statement about the simulator, not about BAR-ATH.
* **The flag means something only where recovery is known.** On the proxy the flag is exact by construction. On
  BAR-ATH it would mark the detector's accepted frames, whose precision is 0.76 at best (test blocks, where the
  ATD saw the ball) and unknown on the hidden frames. The model learns "flagged ball = less trustworthy", at the
  simulated error rates.
* **exp_6's detector quality.** Precision@2 m 0.760, recall 0.230, ĉ 0.312, accepted runs median 3 frames. All
  measured against the ATD (automatic, uncorrected), on test blocks, not on gaps. The measured view inherits
  that uncertainty.
* **Not blind.** ĉ, σ̂ and f̂ were known when the training ranges were drafted (§0.3). The ranges were wide and
  generic (c ≤ 0.6, σ ≤ 1 m, f ≤ 0.5), and the AR(1) row argues against a narrow fit to the noise model.
  The stress view lies partly outside the training range (f 0.53 > 0.5).
* **14 training draws.** Parameters are drawn once per (match, round), so the whole training set holds only 14
  error settings; one draw (J03WQQ round 1) revealed almost nothing (c 0.005).
* **Seven matches, one league, one provider; paired match bootstrap**, so the CIs are wide. The `L_ctrl`
  failures are within one seed's spread, so a rerun with other seeds could flip them in either direction.
* **Label definition.** The model was trained on provider labels, as exp_2's members were. `L_ctrl` is
  evaluated with `L_ctrl`-selected decoders only.

## 9. What this suggests (not done here; each needs its own pre-registration)

1. **The model side is solved on the proxy.** With robustness training, exp_6's recovered ball is no longer
   harmful, and it helps provider by +0.014 [+0.003, +0.027]. The remaining lever is the detector: the robust
   model converts a clean ball into the full clean-ball gain, but turns exp_6's ball into only a third to a half
   of it.
2. **A provider-only adoption** of `exp7_robust_ens3` + recovered ball would have passed here. That would be a
   new hypothesis, and it would need the BAR-ATH plausibility guard and the detector rebuild (~3 h). It is not
   permitted by this pre-registration.
3. **More training draws** (one per batch rather than per (match, round)) and more seeds would likely narrow the
   `target_like` cost and the seed spread. Untested.

## 10. Reproduction and hygiene

* `python prefreeze_checks.py` (before freezing only), `python build_notebooks.py`, then for 00 → 03
  `setsid -f $PY run_notebook.py NB.ipynb > cache/logs/NB.log 2>&1 < /dev/null`. 04 runs only if
  `decision.json` says H1 passed. Interpreter: `.venv-torch`.
* Determinism: exp_1's deterministic training (seeded, `use_deterministic_algorithms`); every view and draw is
  seeded (training 20261108 + 9000 + 1000·round + match index; evaluation exp_6's 20261108 + 7000 + match index).
* Tests: 15 synthetic tests pass (`results/tests.json`).
* Write guard: the file-count + size/mtime digest of exp_0–exp_6 and both `data/` folders is unchanged after 01,
  02, 03 and at the end (`results/guard_01.json`, `guard_02.json`, `guard_03.json`, `guard_final.json`). No
  public ground truth was accessed and no BAR-ATH frame-level output was written.
* Disk: `cache/` holds 1.5 GB (token caches 0.84 GB, predictions, reference predictions; gitignored).

## Files

* `PREREG.md` (frozen; §0 item 5 added before freezing), `REPORT.md`.
* Code: `exp7_common.py`, `exp7_aug.py`, `exp7_fit.py`, `exp7_eval.py`, `exp7_decision.py`, `exp7_target.py`,
  `exp7_report.py`, `prefreeze_checks.py`, `build_notebooks.py`, `run_notebook.py`, `tests/test_exp7.py`.
* Notebooks: `00_protocol`, `01_fits`, `02_evaluation`, `03_decision` (executed, IDSSE only); `04_target` (not run).
* Results: `protocol.json`, `prefreeze_checks.json`, `token_caches.json`, `tests.json`, `rechecks.json`,
  `ref_rows.csv`, `fits.json`, `eval_recheck.json`, `evaluation.json`, `variant_*__sel_*__*.json` (22),
  `decision.json`, `summary.json`, `report_tables.md`, `write_guard_before.json`, `guard_0{1,2,3}.json`,
  `guard_final.json`.
