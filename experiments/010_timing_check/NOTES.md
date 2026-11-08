# 010 timing check (internal, no ground truth)

**Question:** how well do two independent evidence sources agree on WHEN possessions start? Script: `timing_check.py` (default: 008 `v1_ball` vs `v2_tcn_only`, both with the 004 dead-ball overlay and the 006-calibrated lambda).

**Method:** for each possession start in A, distance to the nearest same-team start in B; compared with a circular-shift null (B shifted by 60-600 s, 50 draws). Starts right after a `D` stretch are split out, because both decodes share the same D overlay (identical by construction).

**Result:** 212 vs 208 starts. All starts: 41.5% within 1 s (null 3.3%), median |dt| 1.6 s (null 16.2 s). Starts right after `D` (n=38): 81.6% within 1 s, shared overlay, not evidence. **Genuine H<->A changes (n=174): within 1 s 32.8% (null 3.3%), 2 s 48.3% (6.6%), 5 s 59.8% (16.4%), 10 s 71.8% (32.5%).** Near ball gaps (n=21) 47.6% within 2 s; ball visible (n=153) 48.4%.

**Reading:** ball and team shape agree on about half of possession-change times to within 2 s (about 7x chance) and disagree by more than 5 s on about 40%. So timing is genuinely uncertain, which matches the public-window finding that few transitions match within 0.5-1 s. This measures agreement, not accuracy: shape labels were trained on ball-derived pseudo-labels (partly circular), and both decodes share lambda and the D overlay.

**Next:** use the disagreement as a signal (e.g. widen/average the change time between sources), or treat >5 s disagreements as low-confidence changes.
