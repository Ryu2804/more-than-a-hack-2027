# exp_5 pre-registration — touch-only SetSeq, fresh-seed replication

**Status: DRAFT.**

- Written 2026-11-10 at the end of the exp_4 session, before any exp_5 code or fit exists.
- It becomes binding when `00_protocol.ipynb` freezes it: its SHA-256 and the constants read from code go to `results/protocol.json`, via the `freeze_prereg` / `verify_prereg` pattern of exp_3 and exp_4.
- Every notebook re-checks both and stops on any difference.
- **Before freezing, the executing session may only do two things:**
  - add items to §0, covering anything it sees or changes before freezing;
  - fill the code-derived constants into `protocol.json`.
- Any other edit to §1–§8 is a deviation. It must be written into §0 before freezing.
- After freezing, nothing changes. A needed change is a re-freeze with the old protocol kept as `results/protocol_superseded_N.json`, and it is allowed only while no exp_5 prediction exists.

## 0. What was seen before freezing (this experiment is not blind)

1. **exp_1–exp_3 results**, as listed in exp_4 `PREREG.md` §0:
   - `exp3_ref_ens3` (exp_2's 3-seed ensemble) 0.664 on provider labels and 0.748 on `L_ctrl`;
   - `exp2_aug` 0.657 / 0.735;
   - all on the `target_like` view, 7 held-out IDSSE matches, decoder re-selected per label set.
2. **exp_4 results, all of them** (`exp_4/REPORT.md`, `exp_4/results/decision.json`).
   - **H1 failed.** `exp4_full_ens3` (touch + aux head) vs `exp3_ref_ens3`, `target_like`:
     - provider +0.007 [−0.005, +0.019];
     - `L_ctrl` −0.006 [−0.016, +0.003].
   - **Touch only** (`exp4_touch_ens3`, seeds 20261108 / 09 / 10) vs `exp3_ref_ens3`, a report-only ablation:

     | labels | view | Δ | 95 % CI | wins |
     |---|---|---|---|---|
     | provider | `target_like` | +0.015 | [+0.007, +0.027] | 7/7 |
     | provider | `original` | +0.015 | [+0.006, +0.025] | 6/7 |
     | `L_track(1 s)` | `target_like` | +0.007 | [−0.003, +0.016] | 5/7 |
     | `L_track(1 s)` | `original` | +0.013 | [+0.007, +0.020] | 6/7 |
     | `L_ctrl` | `target_like` | −0.003 | [−0.020, +0.013] | 3/7 |
     | `L_ctrl` | `original` | −0.003 | [−0.018, +0.011] | 3/7 |
     | `L_track(3 s)` | `target_like` | +0.002 | [−0.010, +0.013] | 5/7 |
     | `L_track(3 s)` | `original` | +0.006 | [−0.004, +0.014] | 6/7 |

   - **Touch single seeds** (mean spell-F1):
     - provider, `target_like`: 0.659 / 0.655 / 0.666, spread 0.011;
     - `L_ctrl`, `target_like`: 0.738 / 0.730 / 0.740, spread 0.010.
   - **Generalisation gap.** Touch features shrank the train→held-out NLL gap by ~27 % against exp_2's recipe. Training NLL rose (≈ 0.117 → 0.124) while held-out NLL fell (≈ 0.191 → 0.182). The cause is not understood.
   - **Aux head.** No measurable gain anywhere; it is dropped.
   - **Count gate.** On provider labels every arm, including the reference, fails the count gate on J03WMX (under-count of about 30–37 %).
3. **Why exp_5 exists, and what it can and cannot show.**
   - The touch-only arm was one of **eight** report-only ablation comparisons in exp_4. It is tested here *because it looked best*.
   - exp_5 re-runs that arm with **fresh seeds on the same 7 matches**, so it is a **seed replication**, not an independent-data confirmation.
   - A pass means: the touch-only gain on provider labels is not seed luck.
   - A pass does **not** show that the gain generalises beyond these 7 matches, or to BAR-ATH.
4. **In the drafting session:** no exp_5 code, fit or evaluation exists. The executing session appends here what it sees before freezing: tests, import and timing checks, the reproduction check.
5. **Executing session, 2026-11-10, before freezing (additions only; §1–§8 unchanged).**
   - **Seen.** `PROMPT_exp5.md`, `CLAUDE CODEX.md`, `CHALLENGE.md` § "Possession rules", exp_4 `REPORT.md` / `PREREG.md`, the exp_4 modules and tests, and the exp_1–exp_3 functions they call. No BAR-ATH data, no executed notebook, no `evaluation/gt/`. No exp_5 IDSSE fit or evaluation was run before freezing.
   - **Code.** `exp5_common.py`, `exp5_protocol.py` (fit code only; it is hashed into every exp_5 fingerprint together with `exp1_net`, `exp1_tokens`, `exp2_views`, `exp3_touch`, `exp4_net`, `exp4_aux`, `exp4_protocol`), `exp5_eval.py`, `exp5_diagnostics.py`, `exp5_report.py`, `exp5_target.py`, `tests/test_exp5.py`. exp_4 is used only through its pure pieces (`train_tokens`, `view_tokens`, `globals_context`, `RECIPE`, `_claim`, `exp4_net.train` / `AUX_OFF` / `state_nll`); none of exp_4's arm-level functions is called by exp_5 code.
   - **Tests and import checks.** 16 synthetic tests pass (random arrays; the tiny fits inside them are not IDSSE fits). Importing every exp_5 module leaves the file-count + size/mtime digest of exp_0–exp_4 and `data/` unchanged.
   - **Feasibility of the promised steps (all by import).**
     - Reproduction check (§3): exp_4's cached `exp_4/cache/predictions/exp4_touch_2d42a3706658/outer_J03WMX.npz` exists. exp_4's own fingerprint function, called read-only once from the shell, still returns `2d42a3706658` / `78a2fbc5df96` / `4329112f1f34` for the three touch arms, so exp_4's fit code is unchanged since those fits. `ArmSpec(seed=20261108)` equals exp_4's `exp4_touch` spec exactly (test).
     - Read in place (§2, §6.5): the 84 exp_4 touch `.npz`, the 8 `variant_exp4_touch_ens3__*.json`, exp_4's `ref_train_nll.csv` and `gap_table.csv`, the 16 exp_3 reference result files and exp_3's `reference_rows.csv`. All SHA-256 are frozen in `results/protocol.json` and re-checked by every notebook.
     - Rechecks (§2): notebook 00 recomputes `exp3_ref_ens3` and `exp4_touch_ens3` on provider / `target_like`.
   - **Clarifications (no rule change).**
     - The reproduction check is a fit, so it runs **after** freezing, in notebook 01 before any proxy fit, not before freezing as item 4 suggests. This keeps §6.1 ("frozen before any fit"). Its result goes to `results/repro_check.json` and `REPORT.md`. No timing check was run before freezing; exp_4's ~57 s per fit is the estimate, and exp_5's own times are logged.
     - "Seeds 0–2" in §5 means the three `exp5_touch` members (20261111 / 12 / 13), as exp_4's full-data code used its arm seeds.
     - `exp45_touch_ens6` (members: exp_4's three seeds, then exp_5's) is evaluated on the same 4 label sets × 2 views as `exp5_touch_ens3`, report only. §1 did not list its cells.
     - Gap flag (§6.5): exp_4's code (`exp4_diagnostics.gap_table`) flags on the **`original`**-view gap, although exp_4 `REPORT.md` §7's column header says "(tl)". exp_5 keeps the code's definition for the flag and reports the growth on both views.
     - Seed clause (§5): strict ">" (Δ equal to the spread fails). The spread is the max − min of the three `exp5_touch` single-seed mean spell-F1, provider labels, `target_like`, each with nested re-selection on provider labels.
     - The ctrl non-inferiority bound is strict (> −0.01), as in exp_4.
     - Fit accounting: proxy `.pt` files + 1 reproduction fit + full-data `.pt` + 3 rebuild fits; planned 91 ≤ 100.
     - The "Status: DRAFT" header is left as written (only §0 additions are allowed). This file is binding from the freeze time recorded in `results/protocol.json`.
   - **Hygiene hazard found and handled in code.** `exp1_common` defaults `NUMBA_CACHE_DIR` to `exp_1/cache/numba`; `exp5_common` forces it (and `MPLCONFIGDIR`) into `exp_5/cache/` before any import. Notebook 00 records a write guard (file count + size/mtime digest, no content read) of exp_0–exp_4 and `data/`; notebooks 03 and 04 re-check it.
   - **Resources at freezing.** About 4.2 GB of 19.7 GB RAM available, 5.7 GB disk free, RTX 4060 idle.

## 1. Hypotheses

- **H1 (primary).** SetSeq with exp_3's 4 touch features and no auxiliary head beats the exp_2 3-seed ensemble `exp3_ref_ens3` on provider labels, `target_like` view.
  - Arm: `exp5_touch`, fresh seeds, provider training labels.
  - Ensemble: `exp5_touch_ens3`.
- **Secondary (report only; never drives the decision or adoption):**
  - S1 for `exp5_touch_ens3` on `original` and `target_like`, against provider, `L_track(1 s)`, `L_ctrl` and `L_track(3 s)`, using `exp3_protocol.frame_labels`.
  - Single seeds on provider and `L_ctrl`, both views.
  - **Replication consistency.**
    - Δ(`exp5_touch_ens3`) next to Δ(`exp4_touch_ens3`) per cell.
    - The paired `exp5_touch_ens3` − `exp4_touch_ens3` delta with its CI.
    - `exp4_touch_ens3` is exp_4's cached predictions and stored results, read in place, 0 fits.
  - **Pooled `exp45_touch_ens6`:** the 6-seed mean of exp_4's and exp_5's touch seeds. It is reported, never adopted, because half its seeds were seen.
  - Argmax diagnostic, recall of true possessions < 2 s (IoU 0.5), and the generalisation gap (§6).

## 2. Arms (fixed list; no post-hoc arms)

| arm | touch columns | aux head | seeds | fits |
|---|---|---|---|---|
| `exp5_touch`, `_s1`, `_s2` → `exp5_touch_ens3` | yes | no | **20261111 / 12 / 13** | 84 |
| `exp3_ref_ens3` (reference) | – | – | exp_2's cached `exp2_aug`, `_s1`, `_s2` | 0 |
| `exp2_aug` (reference single seed) | – | – | exp_2 cached | 0 |
| `exp4_touch_ens3` (replication context) | yes | no | exp_4's 20261108 / 09 / 10, cached | 0 |

- **Seeds.** The new seeds appear in no earlier experiment; this was checked with grep over exp_0–exp_4 on 2026-11-10.
- **Ensemble.** Per-step mean of p(H, A, D) and p(boundary) over the 3 seeds (`exp2_protocol.ensemble_predictions`).
- **Reference results** come from exp_3's stored nested results (`exp_3/results/variant_*.json`), read in place.
  - `exp4_touch_ens3` results come from `exp_4/results/variant_exp4_touch_ens3__*.json`, also read in place.
  - The SHA-256 of every reused result file is frozen.
  - Notebook 00 recomputes one `exp3_ref_ens3` cell and one `exp4_touch_ens3` cell and requires equality.
- **Decoder for every arm.** exp_1 main, `D.select_config` with `D.GAMMAS` = {0, 0.5, 1, 2} and λ_team, λ_dead ∈ {0.5, 1, 2, 4, 8, 16}. Decoders are not tuned.

## 3. Fixed design (identical to `exp4_touch`; nothing is tuned)

- **Only the seeds are new.** Every other number is exp_4's `exp4_touch` arm.
- **Recipe.** `exp2_aug`, with `TrainConfig()` unchanged:
  - per training match: round 0 `original`, plus rounds 1–2 `target_aug`;
  - mirror on;
  - training targets are provider state and boundary.
- **Touch features.** exp_3's `exp3_touch` by import, constants unchanged. The caches are exp_3's, read in place.
- **Model and loop.** `exp4_net.SetSeqAux` / `exp4_net.train` with `AUX_OFF`, imported from exp_4 and never copied.
- **No new hyperparameter.**
- **Reproduction check** (1 fit, must pass before the proxy fits).
  - Re-fit the `exp4_touch` outer fold that holds out J03WMX (seed 20261108) through exp_5's code path.
  - Compare it with exp_4's cached prediction for that fold, read in place.
  - Bit-exact is expected. A max |Δp| above 1e-3 stops the run for diagnosis.

## 4. Protocol (exp_3 / exp_4, unchanged)

- **Nesting.** Nested leave-one-match-out over the 7 IDSSE matches.
  - Per seed: 21 pair models for inner validation, plus 7 outer models for held-out scoring.
  - Temperature and decoder are selected on the six inner validation matches only, per label set, with exp_0's objective.
  - The held-out match is scored once with the official `possession_metrics.py`.
- **Views.** `original` and `target_like`; `target_like` is primary.
- **Statistics.** Paired match bootstrap, 1000 draws, seed 20261108, 95 % percentile interval.
- **Evaluation never fits.** With `fit=False`, a missing prediction is an error.

## 5. Decision rule and adoption tree (fixed now)

Δ = spell-F1(`exp5_touch_ens3`) − spell-F1(`exp3_ref_ens3`), provider labels, `target_like`, with nested re-selection on provider labels.

- **H1 passes iff both hold:**
  1. the paired 95 % CI lower bound is **> 0**;
  2. **seed honesty is decisive:** mean Δ exceeds the seed spread (max − min of the single-seed means) of `exp5_touch` on the same cell.
- **Change from exp_4.** In exp_4, seed honesty only labelled the report. It is part of the rule here because exp_5's purpose is to separate signal from seed luck.
- **If H1 passes:**
  - Fit full-data `exp5_touch` seeds 0–2: 3 fits on all 7 matches.
  - Build the BAR-ATH input with exp_3's `exp3_target` code by import: pinned fusion rule plus replay guard, tactical input if fusion is rejected, aggregates only.
  - Write `predictions/exp_5_candidate_provider.csv`. Its decoder is the final config selected on the 7 `target_like` out-of-fold predictions against provider labels.
  - Write `predictions/exp_5_candidate_ctrl.csv` **only if** `L_ctrl` non-inferiority also holds: Δ on `L_ctrl`, `target_like`, has paired CI lower bound > −0.01. Its decoder is selected the same way against `L_ctrl`.
    - Otherwise, no ctrl file is written and exp_3's `L_ctrl` candidate stays the team's option.
  - Every written file must pass the official validator and rebuild byte-identically from 3 independent fits.
- **If H1 fails:** write no exp_5 file. `results/decision.json` says so, and exp_3's two files stay the team's options.
- **Never adopted:** `exp45_touch_ens6`, `exp4_touch_ens3`, any single seed.
- **BAR-ATH plausibility** (dead share, possessions/min) is report-only. No public ground truth, no hand labels, no frame-specific constants.

## 6. Overfitting controls

1. **Freeze first.** This file is frozen before any fit; `freeze_prereg` refuses if any exp_5 prediction exists.
2. **No tuning.** Capacity and training are `TrainConfig()`. No new number except the seeds.
3. **Nested selection only.** The BAR-ATH decoder is selected on the 7 out-of-fold predictions.
4. **One primary comparison.** The arm list is fixed.
5. **Generalisation gap (report only).** exp_4's definition:
   - final training NLL with raw p, T = 1, eval mode, no mirror, all training arrays, provider targets;
   - held-out NLL of the outer and pair predictions;
   - gap = held-out − train, per arm and per family.
   - The reference is the exp_2 recipe; exp_4's `results/ref_train_nll.csv` is read in place.
   - A family whose gap exceeds the reference gap by more than **25 %** is flagged.
6. **Firewall.** exp_4's predictions and results are used only for the reproduction check and the report-only replication rows. They never enter the decision or a selection.

## 7. Budget and order

- **Fit count.**

  | fits | what |
  |---|---|
  | 1 | reproduction check |
  | 84 | proxy fits (3 seeds × 28) |
  | 3 | full-data fits, only if H1 passes |
  | 3 | rebuild fits, only if H1 passes |
  | **91** | total; **cap 100** |

- **Time.** About 57 s per fit on this GPU (exp_4 measured), so the proxy fits take about 80 min.
- **Order.** Reproduction check, then seeds 20261111 → 12 → 13.
- **Parallel work.** An evaluation worker may score finished seeds while fits run. Use at most 3 evaluation processes, because RAM is ~19 GB and exp_4 hit an OOM kill (exit 137).
- **Interpreter and caches.** The shared `.venv-torch`; no new venv, no installs. exp_1–exp_4 caches are read in place.

## 8. Hygiene

- **BAR-ATH access.**
  - BAR-ATH is read only in notebook 04's kernel, and only if H1 passes.
  - That kernel prints aggregates only, and its executed notebook is never opened.
  - Results are read from `results/*.json`.
- **Never:**
  - open the public ground truth (`evaluation/gt/`);
  - use hand labels;
  - write under `data/`;
  - write anything under `exp_0/`–`exp_4/`;
  - export frame-level BAR-ATH output other than the gitignored candidate CSVs.
