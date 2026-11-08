# 002 viterbi

**Hypothesis:** a 2-state Viterbi decoder over soft nearest-player evidence (hold state when the ball is missing) removes the flicker that the majority-vote baseline suffers from, so `spellF1` rises.

**Assets used / dropped:** ATD (players + ball) only. Smart Tagging, homography and video not used yet. Published half boundaries and the 2nd-half kickoff frame used for `X` / `D` outside play.

**Change vs previous experiment:** 001 majority vote over +/-w frames and `D` wherever the ball is missing -> 002 Viterbi H/A (tau 3 m, lambda 50 nats, a priori) + `D` only for ball gaps >= G s (G 8) and the pre-kickoff block.

**Result (public 5 min, pipeline sanity only):** spellF1 `v3_lam25_gap8` 0.389, `v2_lam50_gap8` 0.370, `baseline_smooth50` 0.326, `v1_lam50_hold` 0.320, `v4_lam100_gap8` 0.288. Differences are inside the noise of 17 possessions. Possession count on graded frames: Viterbi 159-319 vs baseline 594.

**Verdict / next:** no claim of improvement. The gap -> `D` rule labels 38% of graded frames `D`, far too much (ball gaps are mostly occlusion). lambda is a prior, not evidence-based. Next (003): real dead-ball detection (Smart Tagging restarts with the corrected `Team` semantics, collective player motion, ball out of bounds).
