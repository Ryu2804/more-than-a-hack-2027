# CLAUDE.md — More than a Hack 2027 · Phase 1 (frame-level possession)

Read this before every task in this repo. Milestones and the exact `/goal` commands are in
`GOAL_PLAN.md`. The literature behind the approach is in the team's Claude Doc
"More than a Hack 2027 — Literature & Methods Brief for Frame-Level Possession".

## The task

For each of the 199,319 frames of the tactical video of FC Barcelona (`H`) vs Athletic Club (`A`),
output `H`, `A`, `D` (dead ball) or `X` (abstain).

- Graded frames: 11185–90651 and 110808–190197. The 2nd half kicks off at 113742, so 110808–113741
  are all `D`.
- Ranked on **spellF1**: possessions are maximal runs of `H` or `A`, matched one-to-one to the truth
  by temporal IoU, averaged over thresholds 0.3–0.7. **macroF1** breaks ties.
- For spellF1 the scorer forward-fills `X` with the previous label in the same half. So `X` means
  "no switch here", not "no answer".
- Deliverables (deadline 17 Nov 2026): `<team_name>.csv`, a public repo that regenerates the
  identical file, and a short technical document.

## Hard rules — breaking one can disqualify us

1. **Data hygiene.** The terms forbid giving the match videos, datasets or extracts to any
   third-party hosted LLM, and you are one. Never bring BAR-ATH data into this conversation:
   - do not `cat`, `head`, `less`, `print` or open anything under `data/` (ATD `.txt`/`.xml`,
     homography CSVs, Smart Tagging XML, `data/cache/`);
   - never open, render or describe images or video frames from the match;
   - never read executed notebooks (`experiments/**/*.ipynb`): their outputs hold match data and
     frames. To see a notebook's code, print a stripped copy:
     `jupyter nbconvert --clear-output --to notebook --stdout <nb>`;
   - never print DataFrames, arrays or per-frame predictions for BAR-ATH.
   Scripts that touch BAR-ATH data print aggregate numbers only (counts, means, scores). Read code
   and docs, not data. If an error would dump values, catch it and print a one-line summary. Open
   external datasets (IDSSE, Metrica sample, SkillCorner) are not restricted.
2. **The public ground truth is not a tuning set.** `evaluation/gt/*public*` is read only in
   milestone M4, once, by `scripts/check_public_alignment.py`, to confirm alignment and format.
   Never use it to train, tune, choose a model, threshold or window, or decide whether a change helps.
   `experiments/common.py`'s `score()` scores on the public window: never call it inside a `/goal`
   loop, and do not run `evaluation/run_evaluation.py` on BAR-ATH outputs before M4.
3. **No hand labels and no frame-specific constants.** Every label comes from code. No rule or
   constant may reference specific BAR-ATH frames, except the published half boundaries
   (`games.half_video_blocks`) and the 2nd-half kick-off frame.
4. **No human-produced data about BAR-ATH** (event data, corrected tracking, annotations, from any
   provider). Data about other matches is fine; record it in `TECHNICAL_DOC.md` (what, source, who
   labelled it, how much).
5. **Frozen judge.** Once the git tag `gate-frozen` exists, do not edit `evaluation/`,
   `scripts/gate.py` or `src/mtah/proxy/` (writing submission CSVs into the gitignored
   `evaluation/results/` is fine). If you think the judge is wrong, stop and say why.
6. **Byte-identical reproduction.** `python scripts/build_submission.py` regenerates the submission.
   Fixed seeds, deterministic operations, pinned `requirements.txt`.
7. **Never commit** `data/`, `external/`, caches, or any frame-level export of BAR-ATH.

## Facts from our EDA (`experiments/00_eda.ipynb`; load BAR-ATH with `experiments/common.py`)

These are aggregates already computed; you do not need to re-derive them from the data.
`common.load_atd_arrays()` returns players (N, 46, 3), ball (N, 3) and the team of each slot, cached
in `experiments/_cache/`.

- Ball present on 59.3% of graded frames. 99.7% of missing-ball frames sit in gaps longer than 5 s
  (mean gap 17 s). The ball rarely vanishes near the image border. Interpolating it will not work.
- The nearest team differs across 40–47% of long gaps, so forward-filling through a gap is wrong
  about half the time.
- 46 track slots, no player identities, median tracklet 13.6 s, about 20.5 objects per graded
  frame. Features must be team-level or permutation-invariant.
- Raw nearest-team labels flip 29.6 times per minute. One spurious flip costs a whole possession on
  spellF1.
- Collective speed falls to 0.6–1.5 m/s before restarts; the visible ball is outside the pitch on
  4.2% of visible frames; object counts drop during goal celebrations.
- Smart Tagging: the four DEFENDING / DEFENSIVE codes are exact mirrors of offensive codes (no extra
  information); every throw-in and kick-off has `Team = N/A`; phases last 15–35 s and are not
  possessions. `frame = round(seconds * 25)`.
- Player speed channel is reliable; the ball's third channel is not speed. Ignore it.
- Broadcast ATD aligns to tactical at +4014 frames (1st half) and +26920 frames (2nd half). It
  supplies the ball on 12.1% of tactical missing-ball frames.
- `submission row = ATD frame_id - 1`. Pitch 105.3 × 68.0 m; coordinates are normalised 0–1.

## Approach, in priority order

1. Per-frame probabilities p(H, A, D) from features that exist in both BAR-ATH and the proxy:
   ball–player geometry when the ball is visible; team-shape features on every frame; restart
   triggers. Smart Tagging exists only for BAR-ATH, so it cannot be trained or validated on the
   proxy. Use it, if at all, as a separate prior with a small weight fixed in advance and written in
   `TECHNICAL_DOC.md`, never tuned.
2. Segment decoder: penalised Viterbi or hidden semi-Markov model, with separate H↔A and alive↔D
   switch penalties and minimum durations. Do not hard-code a minimum above about 1 s.
3. Dead-ball module: restart triggers (kick-off, penalty, goal kick, corner, throw-in; free kick by
   default), as in Vidal-Codina et al. 2022. Output a D probability and the restart team (the player
   closest to the ball in the trigger zone).
4. Broadcast ball fusion using the offsets above.
5. Stretch: a ball-free, permutation-invariant set model for long gaps (BallRadar-style), or a
   linked-HMM label model over noisy rules.

## Proxy benchmark — the only basis for model selection

- **Data:** IDSSE / DFL open matches `J03WPY J03WMX J03WN1 J03WOH J03WOY J03WPF J03WQF`, via kloppy
  (`sportec.load_open_tracking_data(match_id=..., only_alive=False)`), stored in `external/`.
  Labels per frame: `D` when the ball is dead, else the ball-owning team (home → `H`, away → `A`).
  Verify kloppy's field names before relying on them. Optional extra training data: Metrica sample
  games via kloppy's Metrica open-data loader.
- **Degradation** (`src/mtah/proxy/degrade.py`, seeded, frozen with the judge): drop identities;
  cut each player track into tracklets (median about 13.6 s); occasionally drop or duplicate a
  player; drop the ball in long blocks (lengths over 5 s, mean about 17 s, about 40% of frames).
  **Decide what to drop from label-free signals only** (collective speed, crowding near the ball).
  Never use ball status or possession labels to place gaps, or the proxy leaks the answer.
- **Protocol:** leave-one-match-out over the 7 matches. Score each held-out match with
  `evaluation/possession_metrics.py` (import it; do not copy or edit it). Compare against the proxy
  baseline: the starter's nearest-player rule plus a ±50-frame majority vote, run on the same
  degraded data. Use a paired bootstrap for Δ = method − baseline.
- **Report per match:** spellF1, macroF1, predicted and true possession counts.
- **Log** every experiment as one row in `experiments/LOG.md`: date, change, mean spellF1, Δ with
  95% CI, decision (keep or revert).

## Layout

Keep the team's existing convention (`experiments/README.md`): one folder per idea,
`experiments/NNN_name/`, copied from `experiments/_template/`, writing only to its own `out/`. In
`/goal` loops use `run.py` scripts, not notebooks, so results print to the transcript. Fill each
`NOTES.md` with proxy results; the "public 5 min" line stays empty until M4. Reusable code lives in
`src/mtah/`.

```
src/mtah/features.py   src/mtah/model.py   src/mtah/decode.py   src/mtah/deadball.py
src/mtah/proxy/{load_idsse.py, degrade.py, benchmark.py}      # frozen with the judge
scripts/gate.py   scripts/build_submission.py   scripts/check_public_alignment.py
experiments/NNN_name/{run.py, NOTES.md}   experiments/LOG.md (one row per experiment)
TECHNICAL_DOC.md   external/  (gitignored)
```

## Reporting

End every turn with four short lines: milestone, what changed, the last gate output lines, next step.
