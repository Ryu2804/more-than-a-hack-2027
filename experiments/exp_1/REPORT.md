# exp_1 report — SetSeq-BD vs exp_0, rasya, ryu (same frozen proxy)

Date: 2026-11-08. Notebooks `00`–`04` in this folder are executed; numbers below are copied from
`results/proxy_report.json`, `results/target_summary.json`, `results/public_sanity.json`.

## 1. Verdict

* **SetSeq (identity-free player-set encoder + BiGRU, trained on re-corrupted IDSSE) is the best arm
  on the shared proxy: mean spellF1 0.656 vs 0.487 (ryu), 0.458 (exp_0), 0.325 (rasya), 0.186
  (baseline). It wins on all 7 held-out matches against every other arm.**
* The headline novelty I hoped for — **boundary-aware decoding — is not supported**: nested selection
  chose γ = 0 (plain constant-penalty Viterbi) in all 7 folds, and forcing γ > 0 is slightly worse
  (−0.007, CI [−0.011, −0.003]). The boundary head itself is informative (AUC 0.94–0.95) but the way
  it enters the decoder only ever adds switch cost, and the model already under-segments.
* **Corruption resampling + mirror augmentation is supported:** +0.055 [+0.040, +0.073], 7/7 matches.
* exp_0's quality gates: 3 of 4 pass; **the ±30% possession-count gate fails** (worst match J03WMX,
  336 predicted vs 511 true, −34%). The arm is not gate-clean.

## 2. What is new here vs the other experiments

| | exp_1 (this) | exp_0 (jeremy) | ryu (`origin/ryu` bd90a9c) | rasya (012) |
|---|---|---|---|---|
| input | raw player tokens, DeepSets pooling, no identities | 88 hand features | 23 hand team-shape/IFAB features + ball channels | nearest player to ball |
| model | 2-layer BiGRU, 60 s windows, state + boundary heads | LightGBM per frame | dilated TCN, dead + team heads, 3 seeds | rules (touch-hold, dead cues) |
| labels | IDSSE provider frame labels | IDSSE | Metrica + SkillCorner, derived from events | none (thresholds) |
| training data | 3 corruption seeds per match + home/away mirror | 1 frozen corruption | ball-channel masking | – |
| decoder | Viterbi with optional time-varying boundary cost | Viterbi + min duration | 2-state Viterbi, λ calibrated to a possession rate | hold + merge |

exp_ryu's working tree only holds the organisers' baseline; ryu's real pipeline lives on the `ryu`
branch, extracted read-only into `reference/ryu/` and run zero-shot with all of ryu's constants.

## 3. Protocol (frozen before any fit, `results/protocol.json`)

* 7 IDSSE matches degraded by exp_0's degrader. exp_0 is being edited concurrently, so exp_1 pins a
  verbatim snapshot (`judge/`, SHA-256 checked) and copies of the degraded matches, verified bit-exact
  against the degrader.
* Nested leave-one-match-out: for held-out *h*, decoder settings are chosen on the other six matches
  using models trained without *h* and without the validation match (21 pair models + 7 outer models).
* Objective (exp_0's): mean spellF1 − 0.15 × mean excess |count error| above 30%.
* Paired match bootstrap, 1000 draws, seed 20261108, 95% interval. Official `possession_metrics.py`.
* Reference arms re-run on the identical data: baseline, ryu final (zero-shot), rasya C06/C08 (rasya's
  frozen parameters), exp_0 LightGBM (nested, same decoder family).

## 4. Proxy results (7 held-out degraded IDSSE matches)

| arm | mean spellF1 | mean macroF1 | mean / max abs count error | Δ vs baseline [95% CI] |
|---|---|---|---|---|
| **exp_1 SetSeq + Viterbi** (`setseq_bd` ≡ `setseq_const`) | **0.656** | **0.918** | 0.21 / 0.34 | +0.469 [+0.450, +0.487] |
| exp_1 SetSeq, forced γ>0 (post-hoc diagnostic) | 0.649 | 0.918 | 0.24 / 0.36 | +0.463 [+0.446, +0.477] |
| exp_1 SetSeq, argmax (no decoder) | 0.624 | 0.918 | 0.09 / 0.18 | +0.438 [+0.411, +0.460] |
| exp_1 SetSeq, no augmentation | 0.601 | 0.895 | 0.23 / 0.35 | +0.414 [+0.378, +0.444] |
| ryu final (shipped TCN, zero-shot) | 0.487 | 0.879 | 0.46 / 0.53 | +0.301 [+0.271, +0.327] |
| exp_0 LightGBM + Viterbi | 0.458 | 0.868 | 0.34 / 0.43 | +0.272 [+0.249, +0.291] |
| rasya C08 (touch-hold + long-slow dead) | 0.325 | 0.592 | 0.41 / 0.49 | +0.139 [+0.126, +0.152] |
| rasya C06 (touch-hold + out_or_slow2 dead) | 0.322 | 0.634 | 0.36 / 0.46 | +0.136 [+0.128, +0.146] |
| baseline (nearest + ±50 majority) | 0.186 | 0.458 | 0.18 / 0.35 | – |

Main arm vs each other arm (paired, 7 matches): ryu +0.168 [+0.132, +0.202]; exp_0 +0.198
[+0.169, +0.225]; rasya C06 +0.333 [+0.315, +0.349]; rasya C08 +0.331 [+0.309, +0.351]; all 7/7 wins.
Per-match tables: `results/per_match_results.csv`; figure: `results/fig_proxy_comparison.png`.

Ablations (paired): temporal decoder vs argmax +0.032 [+0.022, +0.043] supported; augmentation +0.055
[+0.040, +0.073] supported; set model vs LightGBM with the same decoder +0.198 [+0.169, +0.225]
supported; boundary-aware decoding 0.000 (never selected); forced boundary decoding −0.007 worse.

Every model-based arm under-segments on the proxy (true 370–511 possessions per match). SetSeq argmax
is closest on counts (max 18%) but the decoder trades count accuracy for spellF1.

## 5. BAR-ATH (aggregates only, `results/target_summary.json`)

* Final model: SetSeq trained on all 7 proxy matches (+ augmentation); decoder chosen on the 7
  out-of-fold predictions: T 1.30, λ_team 0.5, λ_dead 0.5, γ 0.
* `predictions/exp_1_setseq_bd.csv`: official validator passes; an independent second fit gives a
  byte-identical file (SHA-256 `b2d127de…6570`).
* Ball coverage on graded frames 0.593 (matches the team EDA's 59.3%, so the loader is aligned);
  2 metadata teams, 24 + 22 slots, none dropped; token cap never hit.
* 437 possessions; graded shares H 0.349, A 0.267, D 0.384.

## 6. Public 5-minute window — single use, after freeze, not used for any choice

| pipeline (source) | public spellF1 | note |
|---|---|---|
| **exp_1 SetSeq** (`results/public_sanity.json`) | **0.582** [0.40, 0.75] | macroF1 0.725, 16 vs 17 possessions |
| rasya `smooth125_dead_final` (`002_possession_dp/NOTES.md`) | 0.522 [0.37, 0.68] (CI90) | rasya's own report |
| rasya `dpL_final` λ 32 (same) | 0.488 [0.33, 0.65] (CI90) | rasya's own report |
| ryu final (`TECHNICAL_DOCUMENT.md`, branch `ryu`) | 0.422 [0.29, 0.57] (5–95%) | macroF1 0.778 |
| `baseline_smooth50` (`EVALUATION.md`) | 0.326 | full-match 0.2433 |

17 possessions: every interval overlaps. This row confirms alignment and format only; it ranks nothing.

## 7. Limitations and threats to validity

* **In-domain advantage.** SetSeq and exp_0 are trained on degraded IDSSE and tested on held-out
  degraded IDSSE; ryu's shipped model is evaluated zero-shot and calibrates λ to 2.97 possessions/min
  (its training sources) while IDSSE runs at about 4.6/min. Part of the +0.168 over ryu is domain fit,
  not architecture. On BAR-ATH every arm is a transfer.
* **Proxy ≠ BAR-ATH.** In the 3 IDSSE matches checked, the ball sits off the pitch on 62–74% of
  visible dead frames, while BAR-ATH shows it outside on 4.2% of visible frames; any dead-ball cue the
  model learned from that does not transfer.
* **Grid edge.** Inner selection picked λ at the lowest grid values (0.5–1.0, and 0.5/0.5 for
  BAR-ATH). The frozen grid was not extended after seeing outer results.
* **Count gate fails** (J03WMX −34%); the model merges short possessions.
* One training seed per fit; 7 matches; exp_0's own nested results were still being regenerated by
  another process during this run, so exp_0 is represented by its frozen-snapshot features + LightGBM.

## 8. Pre-registered follow-up: centred boundary decoding (notebook 05)

Rules and grid frozen in `PREREG_followup.md` (SHA-256 `1d9211c2…` in `results/followup_protocol.json`)
before running. Switch cost `max(0, λ − γ·(log b_t − log mean b))`; λ grid widened to 1/16 … 16; same
cached predictions, same nested protocol. Motivated by the outer results above, so not blind.

| arm | mean spellF1 | γ chosen (7 folds) |
|---|---|---|
| `setseq_const_wide` (γ = 0, wide λ) | 0.6556 | 0 everywhere; fold choices identical to exp_1 |
| `setseq_cbd_wide` (centred, wide λ) | 0.6569 | 0.5 in 4 folds, 0 in 3 |

| pre-registered rule | Δ [95% CI] | wins | result |
|---|---|---|---|
| 1. centring helps (cbd_wide − const_wide) | +0.0013 [−0.0009, +0.0038] | 3 of 7 (3 ties) | not passed |
| 2. grid edge mattered (const_wide − const) | 0.0000 | – | not passed |
| 3. beats frozen exp_1 (cbd_wide − setseq_bd) | +0.0013 [−0.0009, +0.0038] | 3 of 7 | not passed → **no follow-up BAR-ATH CSV** |

Count gate still fails for both arms. Centring makes the boundary head *usable* (it is now picked in
4 folds, versus never), but its effect is within noise. The submission candidate stays
`predictions/exp_1_setseq_bd.csv`; notebook 06 was built but deliberately not run.

## 9. Next steps (not done here)

1. Fair architecture test: train ryu's TCN-on-hand-features on the same degraded IDSSE folds.
2. Seed ensemble (3–5) and a count-aware decoder target to address under-segmentation.
3. More proxy matches (beyond 7) before spending more effort on decoder variants: current effects
   of ±0.005 cannot be resolved.
