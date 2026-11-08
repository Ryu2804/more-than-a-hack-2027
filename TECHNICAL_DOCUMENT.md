# Technical document: frame-level ball possession (More than a Hack 2027, Phase 1)

Team: [team name] (members: [names]). Rename this file to `<team_name>.md` before submitting.

## Approach and why

The deliverable is one label per tactical-video frame (`H`, `A`, `D`, `X`). We treat it as a sequence problem: a three-way state (home possession, away possession, dead ball) that changes slowly, observed through noisy tracking in which the ball is missing on about 41% of graded frames. Two components do the work.

1. **Dead ball (`D`) from a supervised model trained on other matches (trained on a marimo/molab sandbox; no data from this match was uploaded).** A two-head temporal convolutional network (TCN) takes per-frame features that do not need the ball: collective motion (mean speed, fraction of players moving), team shape relative to attack direction, and player-configuration cues that follow the set-piece triggers of Vidal-Codina et al. (corner marks, centre mark, players at or beyond the touchline, players in the goal area, players in their own half). Ball channels (control score, ball speed, ball distance to the lines, nearest-player distance) are added but are **hidden on random stretches during training**, so the model learns to work when the ball is missing. One head predicts dead ball, the other the team in possession.
2. **Possession team from a two-state Viterbi decoder.** Ball evidence is the team of the nearest tracked player, counted only when that player is within 1.0 m of the ball (a possession-zone radius taken from the literature; the ball has to be controlled, not just near). It is combined with the TCN's ball-free team evidence at a small fixed weight (kappa 0.02). Missing evidence leaves the state unchanged, so a possession is held through ball gaps. The switch cost lambda is calibrated so that the number of possessions per minute equals the rate measured on other matches (see below).

Why not a per-frame rule: the provided baseline (`baseline_smooth50`) labels the nearest player's team and smooths it; it fragments possessions around contested balls and labels a dead ball only when the ball is missing (about 25% recall of real dead-ball frames in the public window). Why not a vision model: YOLO26 does find the ball on this camera (2-4 px at 1080p), but projecting detections to the pitch needs the tactical homography, which we did not have (see "assets dropped").

## Assets used and dropped

Used: the tactical automatic tracking data (ATD; players and ball). Dropped or not used: Smart Tagging (tried early as a dead-ball prior; the fixed-length 35 s set-piece windows and the `Team` tag, which names the defender for the "defending" codes, made it a weak and error-prone signal); both videos; the broadcast tracking and homography; the tactical homography (not available to us). Tried and dropped: a YOLO ball detector (experiments/009; YOLOv8 fails, YOLO26 works but needs the homography), an unsupervised HMM trained by EM (experiments/005; learns a switch cost that over-fragments), per-frame gap classification for dead ball (experiments/004; 25% recall).

## Training data

Every labelled dataset used for training comes from **other matches**; nothing from the provided game.
- Metrica Sports public sample data, Sample Games 1, 2 and 3 (tracking at 25 fps plus event data), https://github.com/metrica-sports/sample-data (credit: Metrica Sports). About 430,000 frames of tracking, three full matches (~285,000 from games 1-2). Labels per frame (`H`, `A`, `D`) are **derived algorithmically from the event data** (labelled by the provider's operators, converted by our rule): possession belongs to the team of the latest pass, carry, shot, recovery or set piece (loose balls keep the last team, as `CHALLENGE.md` describes); dead ball runs from a ball-out, foul, card or goal to the next ball-in-play event. The derived dead-ball share differs by match (36%, 31% and 24%), which is real between-match variation (game 3 has quick restarts) and not a label artefact; we checked that treating carries as non-restarts changes nothing.
- The calibration target (3.36 possessions per minute) is the mean possession rate of those three games' derived labels (3.35, 3.14, 3.61).
- Published figures used only as sanity checks: La Liga effective playing time 59.3% (Football Observatory / CIES, 37 competitions), implying about 41% dead ball. Earlier experiments also looked at StatsBomb open data (La Liga 2018/19, 24 matches, 1.96 possessions per minute under a coarser possession definition); the final pipeline does not use it.

## Possession rules implemented

- Dead ball: prior-shifted model probability above 0.5 for at least 2 seconds (the shift adds logit(0.407) - logit(0.303) = +0.457 to the log-odds: the training labels are 30% dead on average but "ball not in play" in La Liga is about 40.7% of match time, per Football Observatory/CIES effective playing time of 59.3%); the 2,934 frames before the second-half kick-off (published in the repository) are `D`.
- Possession changes only with controlled evidence: the nearest player must be within 1.0 m. A loose or contested ball keeps the last team. A change needs enough accumulated evidence to pay the switch cost (lambda about 2.4, equivalent to a couple of seconds of consistent evidence).
- Frames outside the two graded halves are `X`. We never abstain inside a half; the scorer fills an `X` with the last label, so holding a state through a gap is what the decoder already does.

## Handling missing and noisy tracking

The ball is absent on 41% of graded frames and track slots are not players. We therefore (1) use only team-level features, never identities, (2) train the model with the ball hidden on random stretches, (3) let the decoder hold its state when there is no evidence, and (4) read the dead ball almost entirely from player motion and configuration, which exist on every frame.

## Results and how far to trust them

On the public 5 minutes (17 possessions), used only to check the pipeline and never to choose parameters, the official scorer gives macroF1 0.778 (bootstrap 5-95%: 0.56 to 0.87) and spellF1 0.422 (0.29 to 0.57), against 0.621 and 0.326 for `baseline_smooth50`. Possession count 19 (true 17) and mean duration 13.9 s (true 14.2 s); dead-ball precision 80% and recall 49%; alive/dead F1 0.927; team accuracy on frames both call live 92.8% (baseline 85.8%). With 17 possessions the spellF1 difference is inside the scorer's noise estimate, so we do **not** claim a proven improvement on the ranking metric.

Evidence that does not depend on the public window: leave-one-game-out validation on three matches (train on two, test on the third; ball hidden, labels derived from events) gives dead-ball AUC 0.90, 0.91, 0.90 and live-frame team accuracy 84.7%, 83.4%, 84.3%, against 0.87 to 0.88 and 79.5% to 81.3% for per-step gradient boosting on the same features. Over the whole match the predicted dead share is 37.3% (published La Liga about 40.7%; 34.8% before the prior shift).

Sensitivity we want to be open about: an earlier two-game model (before fixing a team-mapping bug and an attack-direction rule that affected the training games) scored macroF1 0.857 with 78% dead-ball recall on the same public window; the unshifted three-game model scored 0.749 (41% recall). The window contains four stoppages (5.5, 20.4, 18.3 and 14.5 s), so those differences amount to one or two stoppages and cannot separate the models. We keep the three-game, prior-shifted model because it is validated on three held-out games and uses the most data. The prior-shift step was added after we observed the unshifted model under-predicting dead ball (including on the public window), so it is a post-hoc correction, justified by an external published figure.

## Limitations

- The ground truth's possession definition is not published; our possessions-per-minute target comes from another provider's events (3.36 per minute over three matches) and could be wrong. Two other external references differ (StatsBomb about 2.0, the public window about 3.4), and the true full-match count may sit anywhere from roughly 210 to 360.
- Training labels are our rough derivation from events on three anonymous matches; the match's own ground truth is derived from corrected tracking, which is a different definition.
- Possession-change timing is imprecise: in the public window only 4 of 20 true transitions are matched within one second, and our changes tend to be early by about half a second on average (median 16 frames).
- Event-derived training labels (24-36% dead) understate how much of a match is dead ball (about 41% in La Liga), which is why the prior shift is needed; the shift value comes from a published average, not from this match.
- The ball is still the missing information. With the tactical homography, a ball detector (YOLO26) would likely help; this is the main thing we would do with more time.
- We saw the README figure of 17 public possessions in 5 minutes before choosing the calibration source. The target itself comes from other matches, and the public window was used only as a check.

## Reproduction

```bash
pip install -r requirements-submission.txt        # Python 3.13, numpy 2.5.0, pandas 3.0.3, torch 2.12.1
# data/tactical/BAR-ATH_tactical_atd.xml and .txt must be in place (see data/README.md)
python experiments/015_final_pipeline/run.py       # loads the saved weights and writes
                                                   # experiments/015_final_pipeline/out/submission.csv
python evaluation/validate_submission.py experiments/015_final_pipeline/out/submission.csv
```
The default run loads the saved ensemble (`experiments/015_final_pipeline/weights/`, trained on other matches only, on a marimo/molab sandbox with `experiments/015_final_pipeline/train_model.py`, 150 s on CUDA, Python 3.13.11, torch 2.11.0+cu130); `--retrain` retrains locally (may differ slightly across hardware, which is why the weights are shipped). On our machine every default run gives a byte-identical file; inference on CPU (`--device cpu`) differs in 2 single-frame dead-ball boundary frames, which the scorer treats as free. Experiment notebooks 002 to 014 in `experiments/` document how each choice was made.

## Tooling and credits

[Team to confirm before submitting: statement on the use of LLM tooling and data handling.] Credits: Metrica Sports sample data; StatsBomb open data (early calibration experiments); Football Observatory / CIES (effective playing time); Vidal-Codina et al., "Automatic event detection in football using tracking data" (possession-zone radii, set-piece triggers); the TCN ball-possession study in Expert Systems with Applications (2023); Ultralytics YOLO26 (feasibility test only).
