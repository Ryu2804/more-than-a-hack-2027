# Technical document - ball possession, BAR-ATH

Rename this file to `<team_name>.md` (or `.pdf`) before uploading, so the organisers can match it to
the submission CSV.

## Approach

The submission labels every graded frame as H (home possession), A (away possession) or D (dead
ball). We treat it as a decoding problem over three states with an explicit "no observation" case,
rather than as frame-by-frame classification followed by smoothing.

1. **Observation.** For every frame with a tracked ball, take the player nearest to the ball in
   pitch metres and use that player's team label. Ball position is converted with the pitch
   dimensions in `games.py`. Where the ATD provides a ball speed we use it; otherwise we derive speed
   from consecutive positions divided by the real frame gap and discard steps across gaps longer
   than 25 frames.
2. **Dead-ball evidence (observed only).** Four components, all computed from assets in the bundle:
   - *Kinematics*: smoothed ball speed below 1.0 m/s for at least 30 frames, or net displacement
     under 1.5 m over one second with at least 80% presence.
   - *Geometry*: ball more than 0.5 m outside the pitch for at least 15 frames.
   - *Restarts*: the first 5 seconds of every SET PIECES instance in Smart Tagging.
   - *Gated long missing*: a ball-missing run of at least 5 seconds, accepted only when a restart
     falls within 3 seconds of the run or the last/first ball position is within 3 m of the
     touchline.
3. **Decoding.** A Viterbi pass per half over H/A/D. Emission costs: observed frame 0 for the
   observed team, 1 for the other team, 1 for D; missing frame 0 for all three states; dead evidence
   0 for D and 1 for H/A. Transitions between different labels cost a switch penalty lambda; staying
   costs 0. Frames outside the two graded halves are forced to D.
4. **Lambda selection and plausible possession band.** The switch penalty was initially selected
   based on the synthetic study (`experiments/002_possession_dp/NOTES.md`) targeting plausible
   possession frequencies. In subsequent cross-match validation on IDSSE (experiments 011-012),
   dynamic programming and cue integration thresholds were validated on independent games,
   ensuring that no private match labels are used during inference.

Why this rather than a classifier: the class is a property of a segment, not of a frame, and 41% of
graded frames have no ball observation. A classifier trained on the provided game is forbidden by
the rules, and a classifier trained elsewhere would not see this tracking's error signature. The DP
expresses the two things we do know: missing is not dead, and possessions are persistent.

## Assets used, and assets dropped

Used from the bundle: tactical ATD tracking (`_atd.txt`, `_atd.xml`), tactical Smart Tagging
(`_smart_tagging.xml`), and the public 5-minute GT for correctness checks only.

Tried and dropped:

- **Ungated long-missing as dead.** It marks 40.6% of graded frames dead and conflates occlusion
  with stoppage. Kept only as a negative control that shows the gate is what changes behaviour.
- **Ball-speed threshold alone.** Provided speed exists on roughly 9% of present-ball frames; the
  derived fallback is coarse, so it is used only together with the stationarity test.
- **`lambda = 4`.** The value recommended by our synthetic study assumes independent label noise. The
  real nearest-player sequence switches team 28.4 times per minute, so lambda 4 follows noise and
  over-fragments (95 possessions in five minutes against 17 in the GT). Replaced by the GT-free rule
  above.
- **Coordinate imputation.** Interpolating the ball through missing frames does not create
  possession evidence and was not used at inference.
- **ByteTrack / BoT-SORT.** The bundle contains point tracking and metadata only: no video, no
  bounding boxes, no detector confidences. Multi-object trackers cannot be applied, and applying a
  tracker to a single point would be meaningless.
- **Homography.** Useful for debugging the pitch frame; we found no possession signal in it.

## Training data

None. No model is trained, no labelled dataset is used, and no hand labels were produced for the
provided game. Every decision (evidence thresholds, lambda, gating) was pre-registered or
derived from cross-dataset transfer validation on IDSSE. The public 5-minute window is
used solely for scorer parity verification and veto checks; model selection is performed on independent external matches.

## Possession rules implemented

- A possession is a maximal run of one team's label. H and A are emitted only while the ball is
  alive and attributed to one team.
- D is emitted for stoppages: detected dead-ball evidence, plus everything outside the two graded
  halves. D is never a possession, and it breaks a run.
- Missing ball is *not* dead. The missing state is carried by the previous label until evidence or
  an observation changes it. This is the main difference from the naive "missing means dead" rule,
  which rewrites whole occlusion windows.
- Loose balls and challenges are not modelled explicitly: whoever is nearest to the tracked ball
  claims it, softened by the switch penalty so that a single noisy frame does not flip possession.
- Restarts (kick-offs, throw-ins, free kicks, corners, goal kicks) are treated as dead for their
  first 5 seconds, then play resumes with the nearest-player observation. Smart Tagging intervals
  are phases, not possessions, so we use only their start, never their extent.

## Limitations

- No full-match ground truth is available to us, so accuracy over the full match is unverified. The
  public window holds 17 possessions, which gives bootstrap intervals roughly 0.3 wide; every
  difference that matters here sits inside one interval.
- The evidence layer, not the DP, is what moves the public score: smoothing plus the same mask
  (`smooth125_dead_final`, 0.5222) is not worse than the DP (`dpL_final`, 0.4875), paired
  difference -0.036 with interval [-0.161, +0.061].
- Team identity switches in the ATD are not repaired. The team label is used as-is, and this is the
  dominant noise source.
- The 5-second set-piece window and the 3 m / 3 s missing gate are constants, not measured dead
  times.
- The DP assumes a single switch penalty for all transitions; it does not model restarts or
  half-time differently from open play.

## Reproduction

From the repository root, with the data bundle at `../data` (or set `MTAH_DATA_DIR`):

```
.venv/bin/python experiments/002_possession_dp/run.py \
  --data-dir ../data --bootstrap 1000 --seed 0 --synthetic-seeds 3 --write-submissions
```

This writes `experiments/002_possession_dp/out/` with the submission CSV, `results.json`,
`results.csv`, `run.log` and the official scorer reports. Score separately with:

```
.venv/bin/python evaluation/run_evaluation.py \
  --results experiments/002_possession_dp/out/submissions \
  --out experiments/002_possession_dp/out/official_reports --bootstrap 1000
```

Versions used: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1. Script sha256 is recorded in
`results/results.json`, together with the sha256 of every input file. The run is deterministic: the
same command reproduces the same numbers.
