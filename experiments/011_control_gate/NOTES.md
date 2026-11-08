# 011 control gate

**Hypothesis:** CHALLENGE.md says possession changes on *controlled* possession, not first touch. Our flips come ~2 s early (public-window delay diagnostic). Counting ball evidence only when the nearest player is within a control radius r should give later/cleaner flips.

**Assets used / dropped:** ATD (team distances), 008 TCN shape probabilities (cache `_cache/p_tcn.npy`), 004 gap dead-ball classifier, 006 external target. No GT used to choose r, no labels, all local.

**Change vs previous experiment:** ball evidence gated to frames with nearest-player distance <= r (r = 1.5 m main, 2.5 m alternative; physical prior, fixed before looking at results); lambda recalibrated to the external target per variant (43.6 ungated, 26.4 at 2.5 m, 18.8 at 1.5 m).

**Result (internal):** gate keeps 24.8% (r 1.5) / 35.1% (r 2.5) of graded frames vs 59.2% ungated; possessions 206-211, median start-to-next-start 22-24 s (external 22.3 s); H<->A change times mostly unchanged (median shift 0.00 s on 145-151 of 180 matched). **Public 5 min (pipeline sanity only):** gate15 / gate25 spellF1 0.529 (macroF1 0.669 / 0.673) vs ungated 0.500 (0.665) vs no-shape gate15 0.385 vs baseline 0.326. Delay diagnostic (7 of 12 H<->A found): median -2.20 s ungated -> -1.60 (r 1.5) / -1.36 (r 2.5); within 2 s 29% -> 57%.

**Verdict / next:** principled, definition-motivated change; direction matches the hypothesis on the delay diagnostic, but n=7 and the spellF1 difference is one possession, so no proven gain. Current candidate: `011 v3_gate15` (r fixed a priori at 1.5 m). Next: single regenerate script + pinned deps, then technical document.
