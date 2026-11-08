# 008 tcn shape

**Hypothesis:** a temporal model over team-shape features (about 25 s of context) predicts possession better than 007's per-frame classifier, especially where the ball is missing.

**Assets used / dropped:** 007 shape-feature cache (ATD only), pseudo-labels from ball control (|c|>0.6), 004 gap dead-ball classifier, 006 external target. No GT, no hand labels. Trained locally on the RTX 4060 (CUDA PyTorch); no cloud, no molab/Kaggle.

**Change vs previous experiment:** gradient boosting per frame -> dilated bidirectional TCN (kernel 5, dilations 1,2,4,8,16,1, 64 ch, ~125 steps at 5 fps), trained with ball-channel masking (50% fully masked, else random chunks), 3-seed ensemble, symmetric H/A views; evidence kappa=0.02 with lambda recalibrated to the external target.

**Result (internal):** cross-half AUC vs ball pseudo-labels, ball hidden: TCN 0.881 / 0.894 vs GBDT 0.831 / 0.859 (accuracy 79.7/84.8% vs 75.2/81.2%); TCN with ball (upper bound) 0.975 / 0.986. Sign agreement TCN vs GBDT on ball-missing frames 75.8%. TCN-only decode 208 possessions, median start-to-next-start 23.5 s (external 22.3 s); labels differ from ball-only on 31.7% of ball-missing frames (ball+TCN k02: 17.9%). **Public 5 min (pipeline sanity only):** ball+TCN k02 spellF1 0.500 (= ball-only), TCN-only 0.462 / macroF1 0.670, ball+GBDT 0.400.

**Verdict / next:** consistent two-way gain in shape evidence quality; no measurable gain on the ranking metric can be claimed (17 possessions). Caveats: easy-frame, ball-derived labels (optimistic AUC). Next: internal boundary-timing check, harder pseudo-labels, local video cues (needs tactical homography, still missing).
