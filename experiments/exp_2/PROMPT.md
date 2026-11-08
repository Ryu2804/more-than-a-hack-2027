# exp_2 session prompt — paste this into a new Claude Code session (cwd: barca/)

You are continuing the team's frame-level possession work (More than a Hack 2027, BAR-ATH).
Build **exp_2** in `exp_jeremy/exp_2/`. Save the notebooks there **first** (unexecuted), then run them.

## Read first (code and docs only)

- `exp_jeremy/CLAUDE CODEX.md` — hard rules. **Data hygiene is absolute**: never cat/print/open BAR-ATH
  data, never read executed notebooks that touch BAR-ATH, print aggregates only.
- `exp_jeremy/exp_1/REPORT.md`, `README.md`, `PREREG_followup.md`, `results/proxy_report.json`,
  `results/followup_decision.json`, and the exp_1 modules (`exp1_*.py`). Reuse them by import
  (add `../exp_1` to `sys.path`); do **not** copy them into exp_2.
- Teammates: rasya = rule pipelines (`exp_rasya/.../experiments/012_spellf1`); ryu = TCN on hand features,
  branch `origin/ryu` (already runnable via `exp_1/exp1_ryu.py`). Do not repeat their experiments.

## Where we stand (exp_1, 7 degraded IDSSE matches, nested leave-one-match-out)

| arm | mean spellF1 |
|---|---|
| exp_1 SetSeq (DeepSets + BiGRU) + constant Viterbi | **0.656** (max count error 34%, gate FAIL) |
| SetSeq argmax (no decoder) | 0.624 (max count error 18%) |
| SetSeq without corruption/mirror augmentation | 0.601 |
| ryu shipped TCN, zero-shot | 0.487 |
| exp_0 LightGBM | 0.458 |
| rasya touch-hold rules | 0.325 |
| baseline | 0.186 |

Public 5-min window: exp_1 0.582 [0.40, 0.75]. **The public check is spent; never open the public GT
again** (`exp_1/results/public_check_started.json` exists).

Open problems, in order of expected payoff:
1. **Transfer risk is unmeasured.** Every exp_1 number is in-domain (train and test on degraded IDSSE).
   IDSSE parks the ball off-pitch during dead time (ball outside on 62–74% of visible dead frames in the
   3 matches checked); BAR-ATH shows ball outside on only 4.2% of visible frames. The model may rely on
   a dead-ball cue that does not exist in BAR-ATH.
2. **Under-segmentation.** All trained arms predict too few possessions (SetSeq −4% to −34%); the
   Viterbi trades count accuracy for spellF1, and the ±30% count gate fails.
3. **Decoder variants are below the noise floor.** Boundary-aware and centred-boundary decoding gave
   ±0.007 with 7 matches; do not spend effort on more decoder tweaks unless (1) and (2) are solved.
4. One training seed per fit; 7 matches only.

## exp_2 hypothesis (one primary, pre-registered)

**H1 — Transfer-robust SetSeq.** Training SetSeq with *target-like observation augmentation* (hide the
ball whenever it is outside the pitch, plus extra ball-hiding inside long dead spells, at rates chosen
to match BAR-ATH aggregate statistics from the team EDA: ball visible 59.3%, ball outside 4.2% of
visible frames) and with *more, more varied training matches* keeps in-domain spellF1 within 0.02 of
exp_1 **and** improves performance when the test matches come from a different source/condition.

Secondary (report, do not let them drive selection):
- **H2 — Count-aware decoding:** a semi-Markov / explicit-duration decoder whose possession and dead-spell
  duration priors are estimated from the training folds' labels (not tuned on held-out matches) brings
  every match within ±30% count error without losing more than 0.01 spellF1 vs the constant decoder.
- **H3 — Seed ensemble:** averaging 3 seeds' probabilities changes spellF1 by more than the seed spread.

## Design requirements

1. **Pre-register before any new fit**: `exp_2/PREREG.md` with hypotheses, arms, grids, decision rules
   (paired 95% bootstrap CI lower bound > 0, 1000 draws, seed 20261108), and the count gate. Store its
   SHA-256 in `exp_2/results/protocol.json`; every notebook verifies it.
2. **Frozen judge:** reuse `exp_1/judge/` + `exp_1/cache/proxy/` through `exp1_common` (hash-checked).
   Evaluation degradation for held-out matches stays exp_0's frozen corruption (round 0). Keep exp_1's
   nested pair/outer protocol and selection objective so numbers are comparable to the table above.
3. **New stress test (the key addition):** a second evaluation view of the same 7 held-out matches,
   `target_like`, where the held-out match's ball is hidden whenever outside the pitch and additionally
   hidden on dead frames chosen *without labels* (e.g. collective-speed-based, like the frozen
   degrader). Fix this transform in PREREG and use it for **every** arm, including exp_1's model
   (re-predict from its cached fits' configuration; retrain only if weights were not saved) and ryu's
   zero-shot arm. Primary metric for H1 = mean spellF1 on `target_like`; secondary = original view.
4. **More training data (optional, gated):** IDSSE has only 7 matches. Before using any other source
   (Metrica sample games, SkillCorner open data, as ryu did), confirm and record its licence and
   labelling provenance in `exp_2/DATA_SOURCES.md`; if a licence is unclear, skip it and say so.
   If used, add a **leave-one-source-out** evaluation (train IDSSE → test the other source and vice
   versa) as the transfer view. Labels must come from the provider or from documented event-derived
   rules; never hand labels.
5. **Label-free BAR-ATH plausibility check** (aggregates only, no GT): predicted dead-ball share,
   possessions per minute, mean possession duration for exp_1 vs exp_2 candidates, compared with
   published priors (≈40.7% dead, team EDA). Report only; never select on it.
6. **Arms (minimum):** `exp1_ref` (exp_1 SetSeq), `exp2_aug` (H1 augmentation, IDSSE only),
   `exp2_aug_multi` (if extra data passes the licence check), `exp2_aug_hsmm` (H2 decoder on the H1
   model), `exp2_aug_ens3` (H3). Same 5 fps tokens and SetSeq architecture unless PREREG says otherwise.
7. **Budget:** SetSeq fit ≈ 45–80 s on the RTX 4060; a full nested arm = 28 fits. Cap total new fits at
   ~150. Cache predictions by code+config fingerprint (`exp1_protocol.ensure_predictions` pattern).
   Run long notebooks in the background and wait for the completion notification; no polling loops.
8. **BAR-ATH candidate** only if the pre-registered adoption rule passes (e.g. H1 CI lower > 0 on
   `target_like` and original-view spellF1 not worse than exp_1 by more than 0.02). Write
   `exp_2/predictions/exp_2_candidate.csv`, validate with the official validator, rebuild byte-identically,
   aggregates only. No public check.

## Environment and housekeeping

- Interpreter: `/home/jeremy-mboe/Documents/Kuliah/Sem3/Machine_Learning/.venv-torch/bin/python`.
  **Do not create a venv.** Disk has ~9 GB free: do not copy exp_1 caches; read them in place.
- Notebooks: build with a `build_notebooks.py` (argument = notebook names, so rebuilding one never wipes
  another's outputs), execute with a safe runner like `exp_1/run_notebook.py` (one-line failure summary).
- `exp_2/.gitignore`: `cache/`, `predictions/`, `kernels/`, `*.error.txt`.
- Unit tests in `exp_2/tests/` for the new transforms (label-independence of the target-like view,
  mirror consistency) and the duration decoder.
- Do not commit or push; the user pushes to branch `jeremy` themselves.

## Done means

1. `exp_2/PREREG.md` hash recorded before any new fit, verified in each notebook.
2. Notebooks `00`…`NN` saved in `exp_2/` and executed without error (IDSSE-only notebooks may be read;
   BAR-ATH notebook outputs are aggregates only and are inspected via results JSON).
3. `exp_2/REPORT.md` with: both evaluation views for every arm, paired deltas vs `exp1_ref` with 95% CI,
   each pre-registered rule marked passed/failed, count-gate status, BAR-ATH aggregates, limitations.
4. Tests pass. Nothing under `data/`, no frame-level BAR-ATH export, no public-GT access.
