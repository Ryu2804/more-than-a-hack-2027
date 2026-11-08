# 004 gap classifier

**Hypothesis:** classify each ball gap (>=2 s) as restart vs occlusion from gap geometry (duration, boundary distance at start/end, ball displacement, player motion), trained on algorithmic pseudo-labels (set-piece tag within +/-5 s). Gaps predicted dead -> `D`. Fixes 002 (38% D) and 003 (2% D).

**Assets used / dropped:** ATD + Smart Tagging (labels only, not features). Video/homography not used.

**Change vs previous experiment:** 003 per-frame dead detector -> gap-level classifier; team decoding unchanged from 002 (tau 3 m, lambda 50).

**Result (internal):** cross-half AUC 0.80 / 0.86 vs pseudo-labels. D share 22.7% of graded frames at theta 0.5 (30.0% at 0.35, 17.6% at 0.65). 191 possessions, mean 25.7 s. **Public 5 min (pipeline sanity only):** macroF1 0.662, spellF1 0.400; 9 predicted vs 17 true possessions.

**Verdict / next:** most plausible dead-ball behaviour so far, but no claim of improvement (public window is noise). Pseudo-label bias: long gaps overlap tag windows by construction. D only inside ball gaps. Next: unbiased labels, justify lambda without the public GT, local video cues.
