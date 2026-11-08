# Writeup: how our possession labeller works

Status as of 2026-11-08. This is a plain-language explanation for the team. It is not the official technical document (`TECHNICAL_DOCUMENT.md`), which still describes an older 3-game version and needs updating.

## 1. The task

For every one of the 199,319 frames of the tactical video of FC Barcelona vs Athletic Club we must output one label:

- `H` Barcelona has the ball
- `A` Athletic has the ball
- `D` dead ball (the game is stopped)
- `X` not graded (before kick-off, half-time, after the final whistle)

The ranking metric is `spellF1`: it rewards getting whole possession spells right (start, end, team), not just frame-by-frame accuracy. `macroF1` is only the tie-break. This is why a model with a high `macroF1` can still have a low `spellF1`: flicker between H and A destroys spells.

Rules we follow: no hand labelling, the public 5-minute ground truth is only a sanity check (not training or tuning), no match data in the public repo.

## 2. The input we have and its main weakness

Tracking data (ATD) gives the position of every tracked player and the ball on each frame. The ball is missing on 41% of graded frames (occlusion, balls in the air, crowded boxes). Players are almost always present. So the design must work when the ball is unknown.

## 3. The method in four steps

**Step 1. Features per frame (`experiments/features.py`).** From player positions we compute 23 player features (team shape, distance to own and opposition goal, collective speed, and triggers from the laws of the game such as a player standing at a corner mark or in the goal area) plus 5 ball features (is the ball seen, how fast, how close to the nearest player). Attack direction comes from where the goalkeepers stand, so nothing is hard-coded to this match.

**Step 2. A small neural network (a temporal convolutional network, "TCN").** It reads about 13 seconds either side of each frame and outputs two numbers: probability that the ball is dead, and probability that the home team is the one in possession. We train 3 copies with different random seeds and average them. Ball features are randomly hidden during training so the network learns to cope when the ball is missing.

**Step 3. Ball-distance evidence.** When the ball is visible and a player is within 0.35 m of it, we count that as strong evidence that the player's team has the ball. Farther than that, we ignore it. The network's team guess is added with a small weight (kappa = 0.02).

**Step 4. Decoding (`experiments/015_final_pipeline/decoder.py`).**
1. Dead ball: frames where the dead-ball probability stays above 0.5 for at least 2 seconds become `D`.
2. A two-state Viterbi decoder picks H or A for the rest. Switching team costs a penalty, called lambda, so the labels do not flicker.
3. Lambda is not hand-picked. We choose it so the number of possessions per minute matches what we see in other matches (about 2.97 per minute, the average of Metrica 3.37 and SkillCorner 2.59).
4. Possession runs shorter than 0.5 s are merged into their neighbours.

One command runs all of it: `experiments/015_final_pipeline/run.py`.

## 4. Training data (other matches only)

Everything is public and none of it is the challenge match.

| Data | What it gives | Licence / credit |
|---|---|---|
| Metrica sample games 1 to 3 | tracking plus events, from which we derive labels | github.com/metrica-sports/sample-data |
| SkillCorner open data, 20 matches | tracking plus phases of play (alive or dead, team in possession) | MIT, credit SkillCorner |
| La Liga ball-not-in-play share 40.7% | a prior for the dead-ball share only | Football Observatory / CIES |

Not used yet: the IDSSE/DFL open set (7 Bundesliga matches, CC-BY 4.0), which has per-frame ball status and possession. We have only listed its files.

All training runs on the marimo/molab sandbox, never on the local machine. Local runs are only inference, decoding and scoring.

## 5. How we measure progress without the answer

The public ground truth covers only 17 possessions, which is too little to measure anything. So we built a held-out-match harness: train on some matches, score on matches the model never saw, with labels derived from their own event or phase data, using the official scorer. A change is adopted only if it gains at least 0.01 mean `spellF1` and wins on a majority of held-out matches.

## 6. Results so far

Held-out matches (the model never saw them):

| Set | Baseline spellF1 | Our model spellF1 |
|---|---|---|
| Metrica | 0.429 | 0.649 |
| SkillCorner | 0.357 | 0.686 |

The model beat the baseline on 23 of 23 held-out matches.

Ablations: training on SkillCorner only is about equal to the full mix; training on Metrica only is poor on SkillCorner (0.476). A higher ball-masking rate (`mask03`) gave only +0.004, so it was not adopted.

Public 5-minute window (a sanity check only, never tuned on): macroF1 0.917, spellF1 0.6815.

## 7. Known problems and honest limits

- **Too few possessions on the public window:** we predict 10, the truth is 17. The labels are probably a bit too sticky. We only change this through choices validated on other matches, not by tuning on the public window.
- **Possession-rate target is uncertain.** Providers define a possession differently (StatsBomb 1.96 per minute, SkillCorner 2.59, Metrica 3.25, public window 3.4). A test showed that a 15% error in this target costs less than 0.01 `spellF1`.
- **Missing tactical homography file.** Without it we cannot place video detections on the pitch, which blocks using a YOLO ball detector to recover the missing ball. A calibration model (PnLCalib) exists but was blocked by an automatic safety check, and we did not work around it.
- **Update (later on 2026-11-08):** the decoder sweep was re-run on the new sandbox. One change passed the adoption rule and is now shipped: team-head weight 0.05 and a 1.0 s sliver merge (23 matches 0.676 to 0.686; 7 IDSSE matches 0.563 to 0.572). Five other ideas were tested and rejected: touch-hold ownership from the other team branch, ball-gap interpolation, last-touch features, a longer-context network, and a change-point output. Details are in `experiments/017_multi_match/NOTES.md`.
- **Where the loss is:** we recover 92% or more of true possessions longer than 8 s, but only 14 to 23% of those under 2 s and about half of those between 2 and 4 s. With true spells under 4 s excluded, spellF1 is about 0.73 on the 23 matches and 0.78 on IDSSE. The challenge match's public window has long spells (mean 14.2 s), so the lower headline numbers probably overstate our loss there.
- **Comparison with the other team branch:** on the 7 IDSSE matches, ours reaches spellF1 0.563 against 0.520 for their decoder (macroF1 0.914 against 0.854). On the public window, 0.681 against 0.400 after fixing a speed-unit bug in their loader.

## 8. To do before submission (Nov 17, 2026)

1. Finish the re-run, apply the adoption rule, and fix the final settings.
2. Retrain the final ensemble on the sandbox with all matches, then rerun `run.py` locally and validate the CSV.
3. Decide on the IDSSE data as extra training matches.
4. Update `TECHNICAL_DOCUMENT.md` and `experiments/README.md`. Fill in the team name and members, and confirm the wording about LLM and data handling against the data terms.
5. Check the repo contains no match data before making it public.
