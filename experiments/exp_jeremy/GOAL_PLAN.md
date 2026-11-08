# GOAL_PLAN.md — milestones and `/goal` commands

Run the milestones in order, one `/goal` at a time, and review the diff between them.

**Setup:** Claude Code v2.1.139 or later (`/goal` needs it). Work on a branch
(`git switch -c goal-run`). Turn on auto mode so goal turns run unattended. Replace `<team_name>`
everywhere with the name from your registration.

## Why there is no fixed F1 target

- The score that ranks us (spellF1 on the private full match) cannot be measured before results day.
- A target on the public 5 minutes would turn `/goal` into the tuning loop RULES.md forbids. The
  published baselines show that window picks the wrong model (smooth125 wins there, smooth50 wins on
  the full match).
- The `/goal` evaluator only reads the transcript and may declare an unreachable absolute number
  "impossible", which ends the goal.

So every goal is gated on things the agent can prove in the transcript:

1. **Relative gain on a frozen proxy:** leave-one-match-out on 7 degraded IDSSE matches, beating the
   proxy baseline by at least +0.05 mean spellF1 with a 95% bootstrap CI that excludes 0. The +0.05
   margin is our judgement: it sits just above the organisers' ~0.04 noise band for one match.
2. **No over-segmentation:** predicted possession count within ±30% of the true count on every
   held-out proxy match.
3. **Deliverables:** valid, byte-identical submission; technical document; rules respected.

The published floor (full-match spellF1 0.2433 for `baseline_smooth50`) is a reference only; we
cannot measure against it.

---

## M0 — build the judge (normal prompt, not `/goal`; you review it)

The agent must not write and grade its own exam inside a goal loop, so the judge is built first and
frozen by you. Paste this in plan mode, approve the plan, then let it run:

```
Read CLAUDE.md, GOAL_PLAN.md, CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md, SUBMISSION.md,
evaluation/possession_metrics.py, starter/baseline.py, experiments/README.md, experiments/common.py
and experiments/_template/. Do not read executed notebooks. Follow every hard rule in CLAUDE.md,
especially data hygiene.

Build only the judge and its data:
1. src/mtah/proxy/load_idsse.py: download the 7 IDSSE matches into external/ (add it to
   .gitignore) and return per-frame H/A/D labels plus team-level tracking. Verify kloppy's field
   names for ball state and ball-owning team; print label shares per match.
2. src/mtah/proxy/degrade.py: the seeded degradation in CLAUDE.md. Place ball gaps from label-free
   signals only. Print aggregate stats of the degraded data next to our EDA targets.
3. src/mtah/proxy/benchmark.py: leave-one-match-out runner that takes any pipeline object with
   fit(train_matches) / predict(match) -> H/A/D codes, scores with evaluation/possession_metrics.py,
   runs the proxy baseline on the same data, and computes a paired bootstrap CI for the spellF1 gain
   (seeded, 1000 resamples).
4. scripts/gate.py with stages m1-m4 exactly as specified in GOAL_PLAN.md ("Gate stages"). Each
   stage prints a short per-match table and ends with one line: GATE <STAGE> PASS or
   GATE <STAGE> FAIL: <reasons>, exit code 0 on pass.
5. experiments/LOG.md with a header row.

Do not build the possession model yet. Finish by running the benchmark on the proxy baseline alone
and showing the table. Do not create the gate-frozen tag.
```

**Then you:** read `scripts/gate.py` and `src/mtah/proxy/`, run `python scripts/gate.py --stage m1`
yourself, commit, and run `git tag gate-frozen`.

### Gate stages (what `scripts/gate.py` must check)

| Stage | Passes when |
| --- | --- |
| `m1` | The pipeline in `src/mtah` (features → model → decoder, not the baseline) runs leave-one-match-out on all 7 proxy matches; two consecutive runs print identical numbers; `python scripts/build_submission.py` writes `evaluation/results/<team_name>.csv` from that pipeline; `python evaluation/validate_submission.py` on it passes. |
| `m2` | Everything in `m1`, plus: mean proxy spellF1 gain over the proxy baseline ≥ +0.05 with 95% CI lower bound > 0; mean macroF1 not below the baseline's by more than 0.01; predicted possession count within ±30% of the truth on every held-out match. Writes the result to `experiments/best.json`. |
| `m3` | Everything in `m2`, plus: `experiments/LOG.md` has a row for the dead-ball module and one for broadcast ball fusion, each with Δ, CI and a keep/revert decision; the current mean spellF1 is not below the one stored in `experiments/best.json` before M3 by more than the CI half-width. |
| `m4` | Everything in `m2`, plus: a fresh virtual environment built from `requirements.txt` runs `build_submission.py` twice and both CSVs have the same SHA-256; `validate_submission.py` passes; `TECHNICAL_DOC.md` has the six sections from SUBMISSION.md; `README.md` states the one reproduce command; `git ls-files data external` lists nothing but `data/README.md`; no file in `src/` or `scripts/` except `check_public_alignment.py` mentions `possession_gt_public`. |

---

## M1 — end-to-end pipeline v1

```
/goal Milestone M1 in GOAL_PLAN.md is done. Proof in the transcript: `python scripts/gate.py --stage m1` was run after the last code change, exited 0, and its final line `GATE M1 PASS` is shown with its per-match table; and `git diff --exit-code gate-frozen -- evaluation scripts/gate.py src/mtah/proxy` exited 0. Build a simple first pipeline in src/mtah: per-frame features that exist in both BAR-ATH and the proxy (ball-player geometry when the ball is visible, team-shape features on every frame; no Smart Tagging in the classifier), a gradient-boosted classifier for p(H,A,D) trained on the degraded proxy matches, and a penalised Viterbi decoder with separate H<->A and alive<->D switch penalties tuned only inside the training folds. Constraints: every hard rule in CLAUDE.md, especially data hygiene and never reading the public ground truth; log each experiment in experiments/LOG.md. End each turn with the turn number and the latest gate lines. Or stop after 25 turns with a status report of what works and what blocks the gate.
```

## M2 — beat the proxy baseline by a real margin

```
/goal Milestone M2 in GOAL_PLAN.md is done. Proof in the transcript: `python scripts/gate.py --stage m2` was run after the last code change, exited 0, and its final line `GATE M2 PASS` is shown with the per-match table, the mean spellF1 gain and its 95% CI; and `git diff --exit-code gate-frozen -- evaluation scripts/gate.py src/mtah/proxy` exited 0. Improve the pipeline in priority order: decoder (duration priors learned from proxy labels, restart-team constraint after D), then features for long ball gaps (team-shape dynamics, context from both gap edges), then calibration. Keep a change only if the leave-one-match-out gain improves; log every experiment, kept or reverted, in experiments/LOG.md. Constraints: every hard rule in CLAUDE.md; never read the public ground truth; never tune on the held-out match. End each turn with the turn number and the latest gate lines. Or stop after 40 turns with a status report: best result so far, which gate check fails, and the most promising next idea.
```

## M3 — dead-ball module and broadcast fusion, kept only on evidence

```
/goal Milestone M3 in GOAL_PLAN.md is done. Proof in the transcript: `python scripts/gate.py --stage m3` was run after the last code change, exited 0, and its final line `GATE M3 PASS` is shown; and `git diff --exit-code gate-frozen -- evaluation scripts/gate.py src/mtah/proxy` exited 0. Add (1) a dead-ball module: restart triggers from player positions (kick-off, penalty, goal kick, corner, throw-in, free kick by default) giving a D probability and the restart team, plus collective-speed and ball-out-of-bounds cues; and (2) broadcast ball fusion on BAR-ATH using the offsets in CLAUDE.md. Evaluate the dead-ball module on the proxy; the broadcast fusion cannot be proxied, so keep it only if it changes no proxy-validated component and log its aggregate effect on BAR-ATH (ball coverage, possession count). Constraints: every hard rule in CLAUDE.md. End each turn with the turn number and the latest gate lines. Or stop after 30 turns with a status report.
```

## M4 — freeze, reproduce, document

```
/goal Milestone M4 in GOAL_PLAN.md is done. Proof in the transcript: `python scripts/gate.py --stage m4` was run after the last code change, exited 0, and its final line `GATE M4 PASS` is shown with both SHA-256 hashes; `python scripts/check_public_alignment.py` was run exactly once and printed only alignment and format checks plus one public-window score marked "sanity check, not used for selection"; and `git diff --exit-code gate-frozen -- evaluation scripts/gate.py src/mtah/proxy` exited 0. TECHNICAL_DOC.md covers: approach and why; assets used and dropped; training data (IDSSE and any other external data: source, who labelled it, how much); possession rules implemented (loose balls, challenges, restarts); limitations; reproduction. No model or parameter may change after the public check runs. Constraints: every hard rule in CLAUDE.md. Or stop after 20 turns with a status report.
```

---

## If a goal stalls or misbehaves

- `/goal` shows status and the evaluator's latest reason; `/goal clear` stops it.
- If the evaluator says a threshold is impossible, do not lower it inside the loop. Review the log
  together and decide as a team whether the margin or the plan should change.
- If anything under `data/` gets printed, stop the goal, clear the session (`/clear`) and tighten
  the script that leaked it.
- Ask the organisers (GitHub issue) whether fitting the degradation to BAR-ATH input statistics is
  acceptable. It uses no labels, but confirm before the technical document claims it.
