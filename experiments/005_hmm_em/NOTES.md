# 005 hmm em

**Hypothesis:** learn the team-switch cost (lambda) and evidence scale from this match's own evidence with unsupervised EM on a 2-state HMM, instead of the hand-set tau=3 m / lambda=50 from 002-004. Also a softer team-control evidence (sum over all players of exp(-d/sigma)).

**Assets used / dropped:** ATD only (+ 004's gap dead-ball classifier unchanged for D). No GT, no labels. All local; molab not used (EM runs in seconds; no match data leaves the machine).

**Change vs previous experiment:** evidence nearest vs soft (sigma 3, 5); lambda from EM (plain) and from EM with power likelihood alpha = 1/tau (tau = integrated autocorrelation time of evidence residuals, 45-90 frames).

**Result (internal):** plain EM lambda ~4.4 (stable across halves) -> ~1,060 possessions, mean 4.7 s. Tempered EM lambda 6.5-7.1 -> 47-95 possessions, mean 52-104 s. Hand-set lambda=50 (004) -> 191, 25.7 s. Soft evidence has fewer ambiguous frames (25.8% vs 32.5%). **Public 5 min (pipeline sanity only):** plain EM spellF1 0.08-0.09; tempered soft3 0.317, others 0.02; 004 ref 0.400.

**Verdict / next:** possession-change rate is not identifiable from frame evidence alone; the two principled fixes bracket the answer by 20-200x. No improvement claimed. Next: set lambda from possession-duration statistics of OTHER matches (allowed by RULES if accessible and documented), never from this match's GT.
