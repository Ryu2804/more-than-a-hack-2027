# 014 supervised dead ball in the decoder

**Hypothesis:** 013's dead-ball probability replaces the 004 gap classifier (24.9% recall) as the source of `D`.

**Assets used / dropped:** 013 probabilities (`_cache/p013.npz`), 008 shape probabilities, ATD team distances, Metrica-derived possession rate (012). External sanity: published La Liga effective playing time 59.3% (Football Observatory, 37 competitions). No GT used to choose parameters.

**Change vs previous experiment:** `D` where p(dead) > 0.5 for >= 2 s (both fixed a priori); gate radius 1.0 m (literature 0.5-1.0 m) and 1.5 m; team evidence from 008 or 013's team head; lambda recalibrated to the Metrica target (344 +/- 32 possessions): 3.8 / 5.9 / 2.4.

**Result (internal):** dead share 42.6% (published ~40.7%); 103 stoppages, median 20.2 s (Metrica 20.8 and 15.2 s). **Public 5 min (pipeline check only):** macroF1 0.833-0.857 (from 0.667); D recall 77.9%, precision 85.2% (from 24.9% / 100%); team accuracy on commonly-live frames 88.3-92.0%; spellF1 0.434-0.516 (noise-level vs 0.450).

**Verdict:** current candidate `v3_r10_013` (chosen on principle). The macroF1 gain is almost entirely the D class. Not claimed: spellF1 improvement, full-match D accuracy.
