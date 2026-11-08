# 006 external calibration

**Hypothesis:** the possession-change rate cannot be learned from this match's frame evidence (005), but it can be taken from OTHER matches (allowed by RULES.md), by choosing lambda so the predicted possession count per minute matches an external reference.

**Assets used / dropped:** ATD + 004 gap dead-ball classifier. External: StatsBomb open data (github.com/statsbomb/open-data), La Liga 2018/19, 24 of 34 matches (random sample, seed 7), aggregate stats only (possessions/minute, start-to-next-start durations). Raw files cached in `experiments/_cache/ext/` (gitignored). Attribution: StatsBomb open data terms require crediting StatsBomb in the technical document. No GT used.

**Change vs previous experiment:** lambda picked by matching an external target (208 +/- 21 possessions for 105.9 graded minutes) instead of hand-set (50) or EM (4.4 / ~7).

**Result (internal):** nearest evidence lambda* 42.8 (band 35.5-52.0), about the old hand-set 50. Soft sigma=3 evidence lambda* 184 (different emission scale). Predicted start-to-next-start percentiles match the external distribution (P50 20.6 vs 22.3 s, P90 62.3 vs 70.2 s). **Public 5 min (pipeline sanity only):** spellF1 0.500 (nearest), 0.486 (soft3), vs 0.400 (004 ref), 0.326 baseline; 11 predicted vs 17 true possessions.

**Verdict / next:** gives lambda a defensible external basis; no improvement claimed (public window is noise). Main risk: StatsBomb possession definition is coarser than this challenge's, so the true count may be higher (public window suggests ~11 predicted vs 17 true). Next: test sensitivity with other external references, improve boundary accuracy (only 2/20 transitions matched on the public window), local video cues.
