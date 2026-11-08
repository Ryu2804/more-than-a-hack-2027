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
