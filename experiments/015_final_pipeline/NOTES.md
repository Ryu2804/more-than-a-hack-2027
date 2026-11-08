# 015 final pipeline (single regenerate command)

`python experiments/015_final_pipeline/run.py` regenerates the submission CSV (`out/submission.csv`) from `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` and the saved ensemble in `weights/`. About 1 minute on an RTX 4060; `--device cpu` works; `--retrain` retrains locally (downloads the public Metrica games into `experiments/_cache/ext_metrica`).

**Training happened on marimo (molab sandbox), as requested.** `train_model.py` is self-contained code that uses only other matches' public data (Metrica sample games 1-3, https://github.com/metrica-sports/sample-data); nothing from this match was uploaded. It ran in the sandbox in 150 s (Python 3.13.11, torch 2.11.0+cu130, CUDA): three-fold leave-one-game-out validation plus the final 3-seed ensemble. The weights (3 x 513 KB), the standardisation constants (`meta.npz`) and the validation (`training_report.json`) were copied back into `weights/`. Two sandbox runs were needed: the first exposed a Game 3 loader bug (see below).

**Bugs found through the validation and fixed:** (1) Game 3's player columns were mapped to teams by the metadata player order instead of the tracking spec's column order (11/11 now; team accuracy on that game rose from 48% = chance to 84%); (2) the attack-direction rule (density peak) was unreliable on 2 of 3 training games, replaced by a goalkeeper-based assignment that is consistent across halves (this match's directions are unchanged); (3) Game 3's second half had no period labels, so its labels were wrong (fixed from the event spans).

**Pipeline:** p(dead) from the TCN, prior-shifted, > 0.5 for >= 2 s -> `D`; ball evidence (nearest-player team within 1.0 m) + ball-free TCN team evidence (kappa 0.02) -> two-state Viterbi; lambda calibrated to the mean possession rate of the three training games (3.36 per minute -> 356 possessions; lambda* 2.43 -> 364); `X` outside the halves; `D` before the second-half kick-off.

**Prior shift (added after seeing the 3-game model under-predict dead ball):** event-derived training labels are 24-36% dead (mean 30.3%) but "ball not in play" in La Liga is ~40.7% of match time (effective playing time 59.3%, Football Observatory/CIES). We add logit(0.407) - logit(0.303) = +0.457 to the model's log-odds. Whole-match dead share: 34.8% unshifted, 37.3% shifted. This step was introduced after observing the unshifted result (including on the public window), so it is disclosed as a post-hoc but external-prior-based correction.

**Validation (leave-one-game-out, ball hidden, labels derived from events):** dead-ball AUC TCN 0.903 / 0.909 / 0.901 vs gradient boosting 0.870 / 0.878 / 0.870 (games 1/2/3); team accuracy on live frames TCN 84.7 / 83.4 / 84.3% vs 79.5 / 79.8 / 81.3%. With the ball available the two are close (gradient boosting slightly better on dead AUC for games 2 and 3).

**Official scorer, public 5 minutes (pipeline check only; bootstrap 1000):** macroF1 0.7775 (5-95%: 0.56-0.87), spellF1 0.4222 (0.29-0.57), 19 possessions vs 17 true, mean duration 13.9 s vs 14.2 s; D F1 0.611 (recall 49.3%, precision 80.3%); alive/dead F1 0.927; team accuracy on frames both call live 92.8%. `baseline_smooth50`: 0.6207 (0.46-0.74) / 0.3256 (0.21-0.45).

**Sensitivity disclosure:** an earlier two-game model (games 1-2 only, before the fixes above) scored macroF1 0.857 with dead recall 77.9% on the same window; the unshifted three-game model scored 0.749 (recall 40.6%). The window has only four stoppages (5.5, 20.4, 18.3 and 14.5 s), so these differences amount to one or two stoppages and cannot separate the models. We keep the three-game, shifted model because it is validated on three held-out games and uses more data, not because of any public-window score.

**Reproducibility:** two GPU runs give a byte-identical file (md5 de4c6251...); `--device cpu` differs in 2 single-frame boundary frames. Retraining on different hardware may give slightly different weights, which is why the trained weights are shipped.

## Ablation: long temporal context (literature-motivated, negative result)

The in-game-status paper (Scientific Reports 2022) used player features from up to +/-80 s around each frame. Our TCN's receptive field is 129 steps at 5 fps (26 s, about +/-13 s). We trained a variant with dilations 1,2,4,8,16,32,64,1 (513 steps = 103 s, about +/-51 s, crop 768) on the same marimo sandbox, same three-fold leave-one-game-out protocol and seeds (`long_context_report.json`).

**Decision rule fixed before running:** adopt only if dead-ball AUC (ball hidden) improves by >= 0.01 on at least two of the three held-out games and mean team accuracy drops by at most 1 point.

**Result:** dead-ball AUC change +0.005 (game 1), -0.008 (game 2), +0.013 (game 3); mean team-accuracy change +0.5 points (game 3 +2.5, games 1 and 2 about -0.5). Only one game improved by >= 0.01, so the rule says **keep the current model**. Longer context did not help consistently with this little training data.
