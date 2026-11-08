# 007 team shape

**Hypothesis:** team-level shape (depth, length, width, compactness, speed, relative to attack direction) carries possession information that does not depend on the ball, so it can supply evidence where the ball is missing (41% of graded frames) and sharpen possession-change timing.

**Assets used / dropped:** ATD only (no ball in the shape model). Pseudo-labels from ball control (|c|>0.6, soft sigma 3 m). 004 gap dead-ball classifier and 006 external target (208 +/- 21 possessions) reused. No GT, no hand labels. All local.

**Change vs previous experiment:** extra team evidence kappa*log p(H) from a gradient-boosting shape classifier (symmetric H/A augmentation, per-half attack direction read from goalkeeper density peak); lambda recalibrated per variant to the external target.

**Result (internal):** attack directions sensible (H right in half 1, left in half 2). Cross-half AUC vs ball pseudo-labels 0.81 / 0.87, accuracy 74% / 82%. Shape-only decode: 210 possessions, median start-to-next-start 22.4 s (ball-only 212 / 20.6, external 22.3). Labels differ from ball-only on 10.3% (k=0.02) / 20.7% (k=0.05) / 33.2% (shape-only) of ball-missing frames. **Public 5 min (pipeline sanity only):** ball+shape k02 spellF1 0.500 (= ball-only 0.500), k05 0.415, shape-only 0.314 (macroF1 0.644 with no ball for team).

**Verdict / next:** shape is a real, ball-free possession signal, but no measurable gain can be claimed (public window is noise). Caveat: trained and validated on easy (clear-control) frames; gap frames are harder, so AUC is optimistic. Next: internal timing check around gaps, harder pseudo-labels, local video cues.
