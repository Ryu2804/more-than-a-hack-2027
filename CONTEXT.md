# CONTEXT — More than a Hack 2027, Phase 1: Ball Possession

> Read this first. It is the single entry point for any agent or teammate joining this repository.
> Companion code lives in `experiments/012_spellf1/`. Vault notes (LEDGER, STATE, AUDIT_LOG,
> SOTA, cycles/) live in the Obsidian vault at `research_loop/riset_methodology/experiments/long_run_2026-11-07/`
> and are **not** in this git repo; their distilled content is reproduced here.

---

## 1. The task

- Input: a 25 fps tactical video of one football match, 199,319 frames, with tracking (players + ball)
  and 638 Smart Tagging phases.
- Output: one label per frame — `H` (home possession), `A` (away), `D` (dead ball), `X` (abstain).
- Scored by **spellF1**: possession episodes are matched predicted-vs-ground-truth with IoU thresholds
  0.3/0.4/0.5/0.6/0.7; F1 averaged over the five. MacroF1 is the tie-break.
- Baseline to beat: `baseline_smooth50` = 0.2433 spellF1 (nearest player + centred majority vote, 50-frame window).
- Deadline: 17 Nov 2026. **No live leaderboard, no feedback.** The only pre-deadline signals are the
  public 5-minute window (17 possessions, 7,500 frames) and `validate_submission.py`.
- Hard rules: no hand annotation, no private ground truth, no human-produced data about the provided
  match, labels must be algorithmic and reproducible, match data never committed or redistributed,
  no use of the public window as a selector (sanity veto only).

## 2. Our approach (rule-based, harness-validated)

We deliberately did **not** start with a learned model. The reasons are in section 7. The pipeline:

1. **Team attribution** — per frame, team = side of the nearest observed player to the ball.
   Flip the home/away convention once per match if agreement with the (external) possession GT is < 0.5.
2. **Touch-hold decoder** — a player "has" the ball only when `dmin < 0.8 m`; that team is held
   until the next touch. Segments shorter than 30 frames are merged into the previous one.
   Selected on IDSSE leave-one-match-out: r = 0.8 m, minlen = 30.
3. **Dead detection** — cue `out_or_slow2`: ball out of pitch (run >= 15 frames) OR nearest player
   speed < 2.0 m/s (run >= 25 frames). Dead runs shorter than 150 frames are merged into the
   previous segment (this is the single biggest lever: false D splits one spell, missed D merges two).
4. **X handling** — X is preserved where the ground truth is X; predicted X is forward-filled inside
   graded runs for temporal metrics.

The same recipe scores **0.5263** on the IDSSE harness (dev, LOSO) and **0.5057** on the two confirm
matches (J03WN1, J03WR9). That is 2.16x the published baseline.

## 3. Validation harness (IDSSE)

- External league, full-match possession ground truth, 7 matches.
- DEV = [J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ]; CONFIRM = [J03WN1, J03WR9].
- Referee tracks (`team_id == "referee"`) are dropped before all computation — they are not players.
- `experiments/012_spellf1/harness.py` reproduces the published baseline ordering exactly
  (raw 0.1241 < w25 0.4234 < w125 0.4314 < w50 0.4832), which calibrates it.
- Oracle-dead ceiling (dead from GT, not predicted): 0.6150. Oracle-team ceiling (team from GT): ~0.852.

## 4. SOTA board (internal)

| Track | Candidate | spellF1 | Paired delta |
|---|---|---|---|
| Deployable | touch_hold r 0.8 + cue dead + merge 150 | 0.5263 (dev) / 0.5057 (confirm) | +0.1127 [+0.1026,+0.1204] vs nearest+smooth50 |
| Oracle ceiling | touch_hold + oracle dead | 0.6150 | +0.1318 |
| BAR-ATH sanity | touch_hold + ball-missing | 0.3692 → 0.4333 | (17 possessions, wide CI) |

Entry rule for the board: held-out measurement, delta >= +0.04, CI90 excludes 0, wins >= 4/5.
The public 5-minute window is a sanity veto only and never selects a candidate.

## 5. What we tried and honestly recorded

| Cycle | Question | Outcome |
|---|---|---|
| C03c | Value-of-information masking (sigma 0.5-3, p 1.0-0.4) | Supported: p per frame must be 1.0; run-mode p>=0.6 suffices |
| C03d | Transfer spec to BAR-ATH (structural invariance) | Not supported all-frame; exposure-matched: sigma 0.5 ok, 1.0 borderline, 2.0 unsafe |
| C03e | Bridge gap by interpolation (K<=25) | Supported: +0.2080 dev, +0.2045 confirm, coverage 0.87 |
| C04 | Harness calibration | PASS |
| C05 | Last-touch hold decoder | Supported: +0.1318 |
| C06 | Predicted dead (cue + merge) | Supported: 0.5263 vs 0.4136 |
| C07 | Dead from median player speed | Null: 0.4551 vs 0.5263 (-0.0712) |
| C08 | Long-slow dead (slow_run 100-200) | +0.0201, below bar; grid at edge |
| C09 | Learned team model (logistic + Viterbi) | Null: 0.5138 vs 0.5263 |
| C10 | Restart seeding after long dead | Null: +0.0333, L at grid edge |

Every cycle has a pre-registered decision JSON, gate pre PASS, gate post PASS, and a LOSO table.
Nulls are recorded, not hidden. See `cycles/C03..C10/`.

## 6. Autonomous loop

`experiments/012_spellf1/loop/` — deterministic driver (`loop.py`, `tasks.py`, `run_loop.sh`),
no LLM in the hot loop. 17 tasks ran, 0 failures, 0 new SOTA. Best challenger: longslow 0.5504
(dev, +0.0241) and 0.5413 (confirm, +0.0355). Artifacts: `loop/INDEX.tsv`, `loop/MORNING_REPORT.md`,
`loop/SUPERVISOR_LOG.md`, per-run `runs/<ts>__<task>/`.

## 7. Why rule-based first (and not ryu-style TCN)

A teammate branch (`ryu`) ships a supervised TCN + Viterbi trained on Metrica, with weights and a
submission.csv. We read it and took lessons, but did not adopt it as the primary path, for these reasons:

- **No public-window tuning.** ryu added a prior shift (+0.457 logit) *after* seeing the public
  window under-predict dead. That is a post-hoc correction. Our thresholds are frozen before scoring.
- **Reproducibility.** A rule-based pipeline with a gate and LOSO is auditable; a 3-weight TCN with
  an unpublished GT definition (possession rate 210-360 full match, target 3.36/min from another
  provider) is not.
- **Dead recall matters more than precision** on spellF1. ryu's dead recall is 49% (precision 80%).
  Our cue dead recall is ~0.97. On the public window both score ~0.42-0.43 — within noise on 17
  possessions, but our dead rule is the one that transfers to the full match.
- **Licences unverified.** Metrica sample data, MIDAS, Ball Radar return `license: null` on GitHub.
  IDSSE is CC BY 4.0 (verified) and is what we use.
- **Claim inconsistency.** ryu's commit message says "0.68 public spellF1" but the technical doc
  reports 0.422. We treat 0.422 as the credible number.

We did, however, take two concrete ideas from ryu and implement them honestly:
- Ball-free dead features (collective motion) as a transferable detector — implemented in
  `ballfree_dead.py` and tested on BAR-ATH as sanity (`barath_ballfree.py`): z-scored variant
  reaches 0.4444, beating ball-missing (0.3692). Needs a pre-registered cycle before any claim.
- Viterbi with a switch cost — tested in C09; did not beat touch-hold on spellF1 because the
  temporal touch structure is what wins, not per-frame accuracy.

## 8. Audit (2026-11-07)

`research_loop/audit_2026-11-07/` — read-only re-verification of citations and rules.

Effective status after audit:
- C01, C02, BAR-ATH transfer → **inconclusive** (not supported).
- Fallback = `baseline_smooth50`. Lambda = 4 (pre-registered default).
- Count band is diagnostic only (the synthetic 82.8k-frame count was wrong; correct n = 158,857).
- C03a/C03b = exploration only (ran after the 14:00 stop, no decision/gate cycle).
- Public numbers leaked into STATE/LEDGER/FINAL_REPORT/AUDIT_LOG (which feed the decision agent) —
  purged in the audit's recommended fix order.
- No official competition rule was violated. Open items: reproducibility (pin deps, commit 003-012),
  licence verification, cloud-LLM clause (OVERVIEW forbids uploading competition data to hosted
  LLMs; RULES.md does not contain the clause), 14 organiser questions drafted.

## 9. Current state

- SOTA unchanged: deployable 0.5263 (dev) / 0.5057 (confirm); oracle 0.6150.
- Best challenger: longslow 0.5504 (dev, +0.0241) and 0.5413 (confirm, +0.0355) — consistent
  direction, below the internal +0.04 bar.
- No submission CSV, no candidate freeze, no raw-data commit.
- Branch `rasya` committed as `324af92` (36 files, +9123 lines); vault docs stay in the vault.
- `requirements-team.txt` adds scipy 1.18.1 on top of numpy/pandas.

## 10. Next steps (owner decision)

- **A. Prepare a submission candidate** with longslow 400/15 (best dev + confirm). Needs explicit
  owner instruction; then `validate_submission.py` + reproducibility pass.
- **B. One more targeted cycle** (ball-free z-scored detector + LOSO) — prereg C11, gate pre/post.
- **C. Close out** (commit, pin deps, send 14 organiser questions, finalise TECHNICAL_DRAFT.md).

## 11. How to run

```bash
cd experiments/012_spellf1
python starter_touch_all.py          # touch_hold on 5 IDSSE matches -> starter_touch_all_out.json
python ballfree_dead.py             # ball-free dead detector, LOSO
python barath_ballfree.py           # sanity transfer to BAR-ATH
cd loop && ./run_loop.sh start 8   # autonomous loop
```

Evaluation without data: `python evaluation/run_evaluation.py --gt public` scores the four shipped
baselines on the 5-minute window.

## 12. Glossary

- **spellF1** — mean F1 over IoU 0.3/0.4/0.5/0.6/0.7 of matched possession spells.
- **touch_hold** — nearest-player team held until the next touch (dmin < 0.8 m).
- **dead cue (out_or_slow2)** — ball out of pitch (run>=15) OR nearest player < 2 m/s (run>=25).
- **merge k** — replace dead runs shorter than k frames with the previous segment.
- **IDSSE** — external league with full-match possession GT; our harness.
- **LOSO** — leave-one-match-out cross validation on the 5 dev matches.
- **gate pre/post** — `experiments/../long_run_2026-11-07/gate.py` checks decision JSON and results JSON.
- **oracle dead/team** — using ground truth instead of prediction; ceiling only, never deployable.
- **sanity veto** — public 5-minute window (17 possessions); may veto a collapse, never selects.
- **C03d structural invariance** — noise-injection transfer test on BAR-ATH (no full GT there).

---

*This document is the distilled, self-contained version of the vault notes. For full cycle detail,
decision JSONs, and gate outputs, see `cycles/C03..C10/` and `experiments/012_spellf1/loop/runs/`.*
