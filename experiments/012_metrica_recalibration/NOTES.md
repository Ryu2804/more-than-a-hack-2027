# 012 metrica recalibration

**Hypothesis:** the 006 calibration target (StatsBomb, 1.96 possessions/min) uses a coarser possession definition than this challenge, so we under-segment (11 predicted vs 17 true on the public window). Metrica's own sample games (same provider as the challenge) give a finer, closer reference.

**Assets used / dropped:** Metrica Sports public sample data, Games 1 and 2 (tracking not used here, event data only), cached in `_cache/ext_metrica/`; acknowledge the source (https://github.com/metrica-sports/sample-data) in the technical document. Game 3 excluded (derived labels lopsided: H 19% / A 71%). Decoder from 011 unchanged (gate 1.5 m, TCN shape evidence kappa 0.02, 004 gap dead-ball classifier). No GT used to set the target.

**Change vs previous experiment:** lambda recalibrated to the Metrica-derived possession rate (3.25/min, SD widened to 0.30 from only two games): target 344 +/- 32 possessions vs 208 before.

**Label rule (algorithmic, from events):** possessor = team of the latest PASS/CARRY/SHOT/RECOVERY/SET PIECE; BALL LOST/CHALLENGE do not change it; dead = BALL OUT / FAULT RECEIVED / CARD / goal until the next ball-in-play event. Gives D 31-36% of frames (public GT ~20%), so the dead-ball rule is rough.

**Result (internal):** Metrica games 3.35 and 3.14 possessions/min, median duration 6.4-7.5 s, ~25% under 3 s. lambda* 5.6 (band 4.5-7.1) vs 18.7 under StatsBomb: 341 possessions, 3.22/min, median start-to-next-start 11.7 s. Public window (pipeline sanity only): 15 predicted vs 17 true possessions (11 before); H<->A changes found within 10 s 9 of 12 (7 before), median delay -1.6 s. spellF1 0.450 (vs 0.529 for the StatsBomb calibration, a <2-possession difference inside the 17-possession noise), macroF1 0.667 (vs 0.669).

**Disclosure:** the README's 17 possessions per 5 minutes (3.4/min) was seen before this experiment; the target comes from other games, and the public figure is only corroboration.

**Verdict / next:** current candidate moves to `v1_metrica_gate15` on definition-fit grounds, not score. Largest remaining risk: the true full-match possession count (208 vs 344 depending on definition). Next: supervised TCN on the Metrica-derived labels (team + dead ball) transferred to this match.
