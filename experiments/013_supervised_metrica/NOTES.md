# 013 supervised model from other matches

**Hypothesis:** the literature (TCN ball-possession paper, Expert Syst. Appl. 2023) shows supervised models beat hand rules by a wide margin on dead ball (89.2% vs 56%) and possession (+7.3 pts). Train a two-head TCN (dead, team) on other matches' tracking with labels derived algorithmically from their event data, then transfer.

**Assets used / dropped:** Metrica Sports sample games 1 and 2 (tracking + events; credit https://github.com/metrica-sports/sample-data in the technical document). Game 3 dropped: derived dead share 10% (others 31-36%), labels unreliable. Features (`experiments/features.py`): IFAB-trigger player-configuration features (corner marks, centre mark, touchline, goal areas, own half), collective motion, team shape vs attack direction, and ball channels (control c, speed, inside-distance, nearest distance) that are MASKED during training so the model works with the ball missing. No labels or GT from this match.

**Result (cross-game, labels derived from events):** dead-ball AUC with ball hidden: TCN 0.862 / 0.889 vs gradient boosting 0.866 / 0.851; with ball: TCN 0.877 / 0.903 vs 0.921 / 0.892. Team accuracy on live frames, ball hidden: TCN 82.1 / 83.1% vs 78.5 / 79.4%.

**Transfer to this match:** predicted dead share 42.9% (published La Liga effective playing time 59.3% => 40.7% dead). Public 5-min window, pipeline check only (threshold 0.5 untouched): AUC 0.973 vs GT dead, recall 80% / precision 85%.

**Verdict:** supervised-from-other-matches is the first approach that clearly works for dead ball. Caveats: derived labels, two training games, public check is 5 minutes.
