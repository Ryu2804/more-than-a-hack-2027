# Research notes: what the literature says about our open problems

Collected with the Firecrawl CLI on 2026-11-08. Sources are public papers and articles about other matches and other providers; no match data was sent anywhere. Raw scrapes are kept outside the repo (scratchpad), not committed.

## Problems this research targets

1. We cannot measure progress (17 public possessions, hidden full ground truth, no hand labels).
2. The possession definition is unclear, so the target possession count is uncertain (208 vs 344 per match).
3. The ball is missing on 41% of graded frames, and dead-ball recall is only about 25%.
4. Possession-change timing is scattered by seconds, usually early.

## Findings, by source

**Opta, "Sequences and Possessions in Football" (theanalyst.com).** A possession is one or more sequences in a row for the same team, and it ends only when the opposition gains control of the ball. A sequence starts with a controlled action on the ball. Tackles and interceptions do not start one unless a controlled action follows. Not every second of ball-in-play time belongs to a team. This is a coarser definition than a maximal run of one team, which explains why StatsBomb-style counts (1.96 per minute) are lower than Metrica-event counts (about 3.25 per minute). It also shows the definition really is provider-specific, so the challenge's own definition (maximal H/A run, ball alive) is the one to follow, and the public ground truth is the only direct evidence of it.

**Vidal-Codina et al., "Automatic event detection in football using tracking data" (arXiv 2202.00804, published in Sports Engineering).** A deterministic decision tree on tracking data, no learning.
- Possession zone (PZ): a circle around each player, radius 0.5 m for one provider and 1 m for two others. The ball inside a player's PZ means that player is in possession. Our control gate used 1.5 m, which is looser than the published values.
- Duel zone: a circle of radius 1 m around the ball; two opponents inside it means a duel, and a duel takes priority over possession.
- A possession gain additionally requires an actual touch: a minimum change in ball direction and ball speed. Proximity alone is not enough, and a ball that stays static while a player walks away is not a loss.
- Ball smoothing: Savitzky-Golay, order 2, window 7 frames (0.25 s at 25 Hz).
- Dead-ball intervals and restarts are detected from player configuration alone, using IFAB-law triggers: kick-off (all players in own half and one near the centre mark), goal kick (a player in own goal area), corner (a player at a corner mark), throw-in (a player at or beyond the sideline), penalty (one player at the goal line, one at the penalty mark, the rest outside the area and 9.15 m away). Player positions exist on every frame even when the ball is missing, so these triggers need no ball.
- Stated limitation: the algorithm needs the ball, and tracking data lack in-play versus dead information, which is critical for set-piece detection.

**Using Temporal Convolutional Networks to estimate ball possession in soccer games (Expert Systems with Applications, 2023).** This is almost our task: classify each time window as dead, home-owned or away-owned from tracking data.
- Trained supervised on 35 games from one league with operator-labelled states, using a TCN per tracked object and a permutation-invariant combination (sum, self-attention or 2D convolution).
- Result: 86.2% possession accuracy (7.3 points above the best rule-based method) and 89.2% dead-versus-alive accuracy (33.2 points above the rule-based state of the art).
- Ablation: ball plus its 5 nearest players performs on par with using everyone; the ball alone performs significantly worse.
- Takeaway: learned models beat hand rules by a wide margin on dead ball, but they used 35 labelled games. We have two (Metrica sample games, labels derived from events).

**Sensors 2026, "Automatic Estimation of Football Possession via Improved YOLOv8 Detection and DBSCAN-Based Team Classification" (PMC12944081).** Broadcast-video possession.
- Ball detector: YOLOv8-P2S3A (a P2 small-object head), 79.4% validation AP for the football.
- Possession rule: nearest player within a distance threshold, with entry and exit hysteresis (T_in, T_out), a minimum switch duration K, and a hold length H for missing detections. This is the structure my gate plus Viterbi decoder approximates, and it confirms the hold-through-missing-ball idea.
- Camera motion is compensated by estimating global motion (optical flow) before tracking.

**Other sources found, not read in depth:** a Lund student thesis on locating the ball from player detections with a graph network (ball position inferred from where players are, relevant to ball gaps); Mills et al. on automatic event detection from broadcast-derived tracking; a review of ball detection techniques in sports; a Hugging Face YOLO football-ball model.

## What changes in our plan

1. Control radius: use 0.5 to 1.0 m (the published range), not 1.5 m, as a prior.
2. Add a touch criterion (change in ball direction or speed) when ball positions are available, instead of proximity alone.
3. Dead ball from player configuration (corner marks, centre mark, goal areas, touchline, stillness) so it works when the ball is missing; learn the combination supervised from other matches rather than hand rules, since the TCN paper shows large gains from learning.
4. Keep the possession-count target uncertainty explicit: Opta/StatsBomb-style definitions give about 2 per minute, Metrica event-derived about 3.25, public window about 3.4.
5. For ball recovery, YOLO26 (COCO) already finds the ball on this camera; the P2 head idea in the Sensors paper matches the unreleased `yolo26-p2` architecture. Both still need the tactical homography.

## Second round (2026-11-08, later): in-game status, calibration, ball recovery

**Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data (Scientific Reports, 2022).** Dead versus live from player positions only, no ball. 102 Bundesliga matches at 25 fps with human-annotated in-game status; features are x, y, velocity and acceleration of all 22 players at each frame plus time-shifted copies at +/-0.4, 0.8, 2, 4, 20, 40 and 80 s (1,320 features); four classical models (logistic regression, decision tree, random forest, AdaBoost). Result: up to 92% frame accuracy on unseen matches (AdaBoost), 81% precision for stoppages longer than 2 s, start-time error at most 2 s for 77% of stoppages and end-time error at most 2 s for 81%; a median filter of 3.64 s turns frame predictions into stoppages. Takeaway for us: dead ball is learnable from player motion alone (consistent with notebook 013), and the authors found long context (tens of seconds) useful, whereas our TCN sees about +/-13 s. We test a ~+/-51 s receptive field (see experiments/015 training report).

**PnLCalib (arXiv 2404.08401, github.com/mguti97/PnLCalib).** Open sports-field registration: HRNet keypoint and line heatmaps, RANSAC + DLT initial camera, then points-and-lines optimisation; per-frame output is a 3x4 projection matrix from pitch coordinates (origin at the centre, 105 x 68 m) to pixels, so the ground homography is three of its columns. Code GPL-2.0-or-later, released weights CC BY 4.0 (SoccerNet-calibration pretrained, optional WorldCup fine-tunes), about 265 MB per weight file (keypoints and lines). SoccerNet's camera-calibration challenge (sn-calibration, arXiv 2410.07401 and the 2023 top-1 write-up) are alternatives. This is the realistic way to get pitch coordinates for ball detections without the tactical homography file, but it means downloading third-party pickled weights and running third-party code: not done without the team's explicit go-ahead (an automatic safety check also blocked it). A self-consistency check is available without any ground truth: project this match's tracked players with the estimated homography and compare with detected people (YOLO).

**Not read in depth:** Roboflow's keypoint-calibration guide, MDPI "Individual Locating of Soccer Players from a Single Moving View" (player localisation with a moving camera), a Sports Engineering 2025 paper on semi-automated last-touch detection for out-of-bounds decisions, and the Royal Society Open Science paper on continuous player tracking from discrete broadcast data.

## Third round (2026-11-08, evening): short spells, ball-free possession, ball recovery

Collected with the Firecrawl CLI. Public papers only; no match data left the machine.

**PathCRF, "Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories" (arXiv 2602.12080, code on GitHub under hyunsungkim-ds/pathcrf).** Trains and tests on the IDSSE/DFL open data we already use (5 train, 1 validation, 1 test match; episodes of continuous play, 10 s windows, 5 fps). It treats the 22 players as a fully connected dynamic graph and selects exactly one edge per step as the current possession state: a self-loop (u,u) means u controls the ball, an edge (u,v) means a pass from u to v is in flight. A masked CRF forbids impossible transitions (only control, kick, receive and out-of-play moves are legal) and Viterbi decoding gives a logically consistent possession path; events are the steps where the edge changes. Test results with player tracking only: edge accuracy about 69.6%, event F1 75.7% (precision 73.2%, recall 78.4%) for the dynamic masked CRF, against 48.4% F1 for a ball-proximity baseline (the "Ball TP" row). The ablation says the masked transitions are what matter: dense CRFs add false transitions and drop precision to 62%. Takeaways for us: (1) possession structure should be modelled as a path through controlling players with legal transitions, not as two free teams; our two-state Viterbi has no notion of a pass in flight, which is where short interceptions get lost; (2) the strongest signal for short spells is the temporal ordering of controls, which proximity alone does not give; (3) it needs a per-player graph network, which is a larger change than anything tested so far. The authors align events with ELASTIC and detect missing ball touches with the Ramer-Douglas-Peucker algorithm on the ball trajectory, so IDSSE labels themselves are partly derived.

**Other leads, not read in depth.** A temporal-graph-learning paper on in-play phases (arXiv 2606.09289); broadcast-derived tracking event detection (Sports Engineering 2025 line of work); ball location from player positions with a graph network (a Lund thesis) and from team formation with a neural network (METU, 2022); BallTrack (MVA 2019) for ball detection in long-shot video; a review of ball detection techniques in sports (PMC12453710). All of these point the same way for our missing-ball problem: the ball position is predictable from player configuration to a coarse level, but a pixel-level detector still needs the pitch homography to be useful here.

**What changes in our plan.** 1) Quantify how much of our short-spell loss is real pass-aware structure versus label noise, by re-deriving IDSSE spells with a minimum control duration and re-scoring (cheap). 2) A per-player controller model with legal transitions is the principled way to recover 2 to 8 s spells, but it is a new model class (set or graph encoder, CRF); worth building only if there is time after the submission checks. 3) Running third-party repository code (PathCRF) requires the team's go-ahead; reimplementing the idea ourselves does not.
