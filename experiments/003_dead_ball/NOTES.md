# 003 dead ball

**Hypothesis:** a dead-ball classifier trained on algorithmic pseudo-labels (Smart Tagging set-piece/kick-off windows) from collective features (player speed, fraction moving, ball out of bounds, ball visibility, ball speed) fixes 002's over-labelled `D` (38% from the gap rule).

**Assets used / dropped:** ATD + Smart Tagging (set pieces, goals). Video, homography not used.

**Change vs previous experiment:** 002 gap>=8 s -> `D`; 003 logistic regression -> prior-corrected log-odds -> 3-state Viterbi (H/A/D; lambda 50, mu 30, kappa 0.04, a priori).

**Result (internal):** cross-half AUC vs pseudo-labels 0.75 / 0.77. Decoded `D` share 2.1% of graded frames (public-window GT is ~20%), insensitive to kappa and mu. **Public 5 min (pipeline sanity only):** spellF1 0.320, macroF1 0.511 (D class nearly empty, 8 predicted vs 17 true possessions).

**Verdict / next:** detector too weak; no improvement over 002 v2. Known artefact: D wins ties when team evidence is ~0 (v5_nodead had 6.6% D with no dead evidence). Next: better pseudo-label alignment, ball out-of-bounds with margin/duration, local video cues.
