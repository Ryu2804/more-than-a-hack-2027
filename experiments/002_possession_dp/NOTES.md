# 002 possession DP

**Hypothesis:** a missing-aware 3-state (H/A/D) DP per half, fed with observed-only dead-ball
evidence (ball kinematics, pitch geometry, Smart Tagging restarts, gated long-missing runs), beats
the strongest smoothing baseline `baseline_smooth125` on spellF1, and the gating matters compared
with the ungated `long_missing` mask used in the first smoke test.

**Assets used / dropped:**
Used: tactical ATD tracking, ATD XML team labels, Smart Tagging restarts, public GT (5-minute window)
for correctness checks only.
Dropped: `long_missing` without a gate (40.6% of graded frames marked dead; kept only as a negative
control), `lambda = 4` from the earlier synthetic note (does not transfer to real tracking),
homography (no possession signal), ByteTrack/BoT-SORT (no video and no bounding boxes in the bundle).

**Change vs previous experiment:** replaces the v1 smoke script. Fixes the ball-speed fallback
(divide by the real frame gap, refuse gaps over 25 frames), adds gated evidence, replaces the
public-window lambda sweep with a ground-truth-free full-match count rule, adds paired bootstrap
differences, synthetic autocorrelation harness, and a masking probe.

**Result (public 5 min, sanity only):**

| arm | spellF1 | CI90 | macroF1 | poss | D F1 |
|---|---|---|---|---|---|
| baseline_smooth50 | 0.3256 | 0.2140-0.4465 | 0.6207 | 26 | 0.323 |
| baseline_smooth125 | 0.4242 | 0.2667-0.5939 | 0.5812 | 16 | 0.146 |
| smooth125_dead_final | 0.5222 | 0.3667-0.6778 | 0.6850 | 19 | 0.455 |
| dp4_final | 0.0786 | 0.0393-0.1286 | 0.6134 | 95 | 0.455 |
| dpL_final (lambda 32) | 0.4875 | 0.3250-0.6500 | 0.6803 | 15 | 0.438 |

Paired: `dpL_final - baseline_smooth125` = +0.063 [-0.034, +0.170];
`dpL_final - smooth125_dead_final` = -0.036 [-0.161, +0.061].
GT-free full-match selection: lambda 32, 280 predicted possessions (count band centre 270.4).

**Verdict / next:** inconclusive. The evidence mask is the active ingredient; the DP does not
demonstrate added value over the same mask applied to smoothing, and `lambda = 4` is not supported
on real tracking. Carry forward `dpL_final` as the GT-free submission candidate and
`smooth125_dead_final` as the comparator/fallback. Next: full-match GT comparison when available,
and calibrating the synthetic confusion generator to the measured 28.4 nearest-player switches per
minute. Details, failure modes and compute: `REPORT.md`.
