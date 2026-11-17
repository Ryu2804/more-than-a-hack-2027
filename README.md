<h2 align="center">PASS: Possession-Aware Spell Segmentation with Ensembled Temporal Convolutions and Rate-Calibrated Viterbi Decoding</h2>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"></a>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c.svg" alt="PyTorch 2.13.0"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-green.svg" alt="MIT License (code only)"></a>
  <img src="https://img.shields.io/badge/Challenge-More%20than%20a%20Hack%202027-a50044.svg" alt="More than a Hack 2027">
  <img src="https://img.shields.io/badge/Public%20spellF1-0.772-orange.svg" alt="Public spellF1 0.772">
  <img src="https://img.shields.io/badge/Method-PP--TCN%20%2B%20BiGRU-black.svg" alt="PP-TCN + BiGRU">
</p>

<p align="center">
  <b>More than a Hack 2027, Phase 1</b> · Submission of carried by admin, Institut Teknologi Sepuluh Nopember (Indonesia)<br>
  Challenge organised by <a href="https://metrica-sports.com">Metrica Sports</a>, the <a href="https://www.fcbarcelona.com">FC Barcelona Innovation Hub</a> and the <a href="https://www.mwcbarcelona.com">Mobile World Capital Barcelona</a>.
</p>

<table align="center">
  <tr>
    <th>Name</th>
    <th>Campus</th>
  </tr>
  <tr>
    <td>Rasya Arya Ramadhan</td>
    <td>Sepuluh Nopember Institute of Technology</td>
  </tr>
  <tr>
    <td>Benedictus Ryu Gunawan</td>
    <td>Sepuluh Nopember Institute of Technology</td>
  </tr>
  <tr>
    <td>Ahmad Naufal Farras</td>
    <td>Sepuluh Nopember Institute of Technology</td>
  </tr>
  <tr>
    <td>Jeremy Mattathis Mboe</td>
    <td>Sepuluh Nopember Institute of Technology</td>
  </tr>
  <tr>
    <td>Jylan Annisa Mumtaza Syidana</td>
    <td>Sepuluh Nopember Institute of Technology</td>
  </tr>
</table>

| Deliverable | File |
|---|---|
| Possession file (scored) | [`carried_by_admin.csv`](carried_by_admin.csv): 199,319 rows, SHA-256 `850e6d25d96ea051d0f746548ab59de9b4c04fdededca53fff16fbe9c2053b5c`. The final method **trained from scratch** by [`E06_wave5.ipynb`](experiments/E06_wave5/E06_wave5.ipynb) ([§5.4](#54-from-scratch-reproduction)) |
| Technical document | [`carried_by_admin.pdf`](carried_by_admin.pdf) (IEEE format, source [`docs/report/`](docs/report/)) |
| Code | this repository; the final method lives in [`experiments/E06_wave5/`](experiments/E06_wave5/) |
| Web dashboard | [`web/`](web/): English match review, synchronized local video and saved-method replay; [live application](https://possession-match-review.vercel.app) |

## Demo

<p align="center"><a href="docs/demo.webm"><img src="docs/demo.gif" alt="Walkthrough of the possession review web app: clip review, match possession and results explained" width="960"></a></p>

[Play/download the demo video (WebM)](docs/demo.webm) · [GIF fallback](docs/demo.gif)

---

## Abstract

Every one of the 199,319 frames of a full LaLiga match (FC Barcelona vs Athletic Club) must be labelled home possession (`H`), away possession (`A`), dead ball (`D`) or no answer (`X`), using only automatic, uncorrected tracking in which the ball is missing on about 41% of graded frames. Because the ranking metric (spellF1) scores whole possessions rather than frames, we treat the task as **temporal segmentation**. The final method combines a three-seed temporal convolutional network with an attention-pooled player-set branch (PP-TCN), trained on 19 external matches with the state loss doubled on complete possessions shorter than 4 s, with a set-sequence BiGRU (75/25 joint mixture). A decoder then masks dead balls, runs an interval-reset two-state Viterbi pass whose switch cost is calibrated to the training possession rate, and merges sub-second runs. The configuration was selected by a pre-registered, provider-balanced rule over 96 configurations: external development spellF1 **0.670** (+0.014 over the previous campaign). The submitted file was produced by retraining both networks from scratch on public data and rebuilding every target input from the raw match bundle in a single notebook. On the 5-minute public ground truth it scores spellF1 **0.772** and macroF1 **0.901**, against 0.326 / 0.621 for the official baseline. The full-match score is unknown. Explainability analyses (stage ablations, Integrated Gradients, receptive field, attention, occlusion, seed uncertainty) show that the network decides mostly from the local ±7 s, relies on team shape and the player set more than on the ball, and that the remaining errors are short contested exchanges and late stoppages rather than ball loss. Fourteen experiments, including six negative or non-promoted ones, are documented with their decision rules and evidence.

## Contents

- [Abstract](#abstract)
- [Contents](#contents)
- [1. Problem and metric](#1-problem-and-metric)
- [2. Data analysis: what makes it hard](#2-data-analysis-what-makes-it-hard)
- [3. Method](#3-method)
- [4. Training data and protocol](#4-training-data-and-protocol)
- [5. Results](#5-results)
  - [5.1 Public window (diagnostic)](#51-public-window-diagnostic)
  - [5.2 Selection on external data](#52-selection-on-external-data)
  - [5.3 Where the method succeeds and fails](#53-where-the-method-succeeds-and-fails)
  - [5.4 From-scratch reproduction](#54-from-scratch-reproduction)
  - [5.5 Explainability](#55-explainability)
- [6. Research history](#6-research-history)
- [7. Limitations and threats to validity](#7-limitations-and-threats-to-validity)
- [8. Reproducibility](#8-reproducibility)
- [9. Repository layout](#9-repository-layout)
- [10. Challenge documents, licence and citation](#10-challenge-documents-licence-and-citation)
- [11. Web review application](#11-web-review-application)

---

## 1. Problem and metric

| | |
|---|---|
| **Input** | tactical video (reference timeline), broadcast video (not synchronised), automatic tracking (ATD) with the ball, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X` |
| **Graded frames** | 158,857 (two halves); everything else is free `X` |
| **Public GT** | 7,500 frames, 17 possessions, used to check the pipeline and never for tuning ([RULES.md](RULES.md)) |
| **Private GT** | the full match, used for the official ranking |

A *possession* (spell) is a maximal run of one team with the ball alive. Predicted and true spells of the same team are matched one-to-one; the F1 of that matching is averaged over five IoU thresholds:

$$
\mathrm{IoU}(p,g)=\frac{|p\cap g|}{|p\cup g|},\qquad
\mathrm{spellF1}=\frac{1}{5}\sum_{\tau\in\{0.3,0.4,0.5,0.6,0.7\}}\mathrm{F1}_\tau .
$$

Every spell counts once, so a one-frame flicker inside a long possession costs a whole spell, and a missed 2-second possession costs as much as a missed 40-second one. **macroF1** (mean per-frame F1 over H/A/D) is reported as the secondary metric. The two disagree strongly: the four official baselines differ only in a smoothing window, and spellF1 moves eleven times as much as macroF1 ([EVALUATION.md](EVALUATION.md); [§5.1](#51-public-window-diagnostic)).

## 2. Data analysis: what makes it hard

From [E00](experiments/E00_eda/) (computed on ATD and Smart Tagging only, without the GT):

| Property | Value | Design consequence |
|---|---|---|
| Ball visible on graded frames | 59.3% | the ball is missing on 40.7% of frames → use player configuration, not only the ball |
| Ball-missing gaps | 150; median 10.7 s, P90 33.8 s, max 221 s | 85.2% of missing frames lie in gaps > 10 s → holding the last label is unsafe; long temporal context needed |
| P(ball lost within 1 s \| visible) | 3.6% | ball loss is a persistent state, not per-frame noise |
| Track slots / tracked players per frame | 46 slots (24 H, 22 A) / mean 20.5 | slots are not players → team-level or permutation-invariant features |
| Tracklet length | median 13.6 s, P90 201 s | per-slot features are unstable |
| Nearest-player distance to ball | median 1.9 m (P90 7.0 m) | proximity is informative when the ball is visible |
| Home/away nearest-distance margin < 2 m | 32.5% of visible frames | "nearest player" is ambiguous a third of the time → temporal evidence needed |
| Smart Tagging `Team` agreement with geometry | 53.7% raw, 63.4% after flipping "defending" codes | tag semantics are a trap; not used as input |
| Broadcast homography valid | 65.8% | broadcast not synchronised; not used |

## 3. Method

![Pipeline of the final method](image/fig01_pipeline.png)

*Figure 1. The final pipeline, `farras_wave5_short_gru25_dead45`. The model works offline at 5 fps; labels are expanded to the 25-fps tactical timeline.*

**Observations.** For each of two mirrored team views: 23 team-shape channels (depth, length, width, compactness, speeds, own-goal-area counts …), a validity flag, 5 ball channels (control, visibility, log speed, distance inside the pitch, log nearest distance) and an unordered set of up to 48 player vectors (10 features: same team, attacking x/y and velocity, speed, ball distance, ball visible, closing speed, nearest-opponent distance).

**Short-loss PP-TCN (weight 0.75).** A 1×1 input projection to 64 channels, then six residual blocks (GroupNorm → dilated Conv1d, kernel 5, dilations 1/2/4/8/16/1 → GELU → dropout 0.15). Receptive field: 129 samples (±12.8 s). The player set is encoded by a two-layer MLP and pooled by softmax attention plus a masked max (permutation-invariant), then added to the temporal stream. Two logits per sample: dead ball, and "my team has the ball". Both are trained with binary cross-entropy, and the **team-state loss is multiplied by 2 on complete live spells shorter than 4 s** (duration measured on the full half, so crop fragments are not up-weighted). Recipe: AdamW (lr 0.002, weight decay 0.01), OneCycle, 1,500 updates of batch 16 on 384-sample crops, ball masking p = 0.3, seeds 0/1/2 averaged over both views. 135,219 parameters per network.

**Set-sequence BiGRU (weight 0.25).** Player-set encoder followed by a two-layer bidirectional GRU (hidden 32, window 300, 8 epochs, seed 2027), with the recipe of [E04](experiments/E04_wave3/): 5 IDSSE matches × 3 corruption rounds, CPU, one thread. It is trained from scratch for the submitted file.

**Fusion.** $p = 0.75\,p_\text{TCN} + 0.25\,p_\text{GRU}$ on the joint $(p_H, p_A, p_D)$, then renormalised.

**Decoder `dead45`.**
1. *Dead mask:* a sample is dead if $p_D > 0.45$ within a run lasting ≥ 1 s.
2. *Team evidence:* $\ell_H=\kappa\log p_T$, $\ell_A=\kappa\log(1-p_T)$ with $\kappa=0.05$. When the ball is visible and a player is within 0.35 m, add $-\log(1+e^{-m})$ to $\ell_H$ and $-\log(1+e^{m})$ to $\ell_A$, where $m=(d_A-d_H)/\tau$ and $\tau=3$.
3. *Viterbi:* a two-state H/A Viterbi pass with switch cost λ, **reset in every live interval** so that no possession crosses a stoppage.
4. *Rate calibration:* λ is chosen from an 18-value grid so the decoded possession count (after merging) matches $\hat n=\rho\cdot T_\text{graded}$. Here $\rho$ is the possession rate per minute estimated from the training matches only, provider-balanced. No target labels are used.
5. *Merge:* live runs shorter than 1 s are absorbed into the previous run.

There is no target-specific rule (no forced dead balls or kickoff handling). Video, Smart Tagging, homography and any human labels of the target match are **not** inputs. Full specification: [docs/FINAL_METHOD.md](docs/FINAL_METHOD.md).

## 4. Training data and protocol

No labelled data from the provided match is used for training.

| Source | Matches | Labels | Licence | Role in the final method |
|---|---:|---|---|---|
| [IDSSE](https://doi.org/10.1038/s41597-025-04505-y) (Bassek et al. 2025) | 7 (5 train, 2 holdout) | provider possession labels | CC BY 4.0 | PP-TCN and BiGRU training; holdout |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | 20 (14 train, 6 holdout) | provider phases of play (proxy) | MIT | PP-TCN training; holdout + ball-drop stress |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | 3 (zero-shot) | event-derived proxy | no explicit licence found | development selection (Games 1/2), evaluation (Game 3); never training |

**Protocol.** There are six grouped folds, and all corruption rounds of a match leave the training set together. Normalisation and decoder statistics come from the training fold only. The objective is the **equal-provider mean spellF1**: conditions are averaged within a match, then matches within a provider, then the three providers with equal weight. It covers 23 development scenarios from 21 matches. The adoption rule was hashed before fitting: a challenger replaces the control only if it gains ≥ 0.01, wins ≥ 2 of 3 providers, and loses ≤ 0.03 on any provider.

## 5. Results

### 5.1 Public window (diagnostic)

![Public window timeline](image/fig02_public_timeline.png)

*Figure 2. Labels on the 5-minute public window. The baseline fragments long possessions; the final method recovers them with boundaries within a few frames, but merges several brief exchanges between minutes 10 and 12.*

![Public scoreboard](image/fig03_public_scoreboard.png)

*Figure 3. Official scorer on the public GT. Left: spellF1 (circles) and macroF1 (squares). Right: number of predicted possessions against the 17 in the GT.*

| Submission | spellF1 | macroF1 | Cohen's κ | Possessions | Mean duration (s) | Transition F1 (±25 fr.) |
|---|---:|---:|---:|---:|---:|---:|
| **Final method, submitted (retrained)** | **0.772** | **0.901** | **0.867** | 12 | 21.5 | **0.588** |
| Baseline ±125 | 0.424 | 0.581 | 0.536 | 16 | 18.5 | 0.167 |
| Baseline ±50 (official floor) | 0.326 | 0.621 | 0.520 | 26 | 10.8 | 0.082 |
| Baseline ±25 | 0.287 | 0.620 | 0.488 | 36 | 7.6 | 0.203 |
| Baseline raw | 0.038 | 0.597 | 0.428 | 131 | 2.0 | 0.141 |

GT: 17 possessions, mean 14.2 s. These numbers were recomputed with [`evaluation/run_evaluation.py`](evaluation/run_evaluation.py). With 17 possessions, one possession moves spellF1 by about 0.06; this is a pipeline check, not evidence of private performance.

### 5.2 Selection on external data

![Development grid](image/fig04_development_grid.png)

*Figure 4. All 96 configurations of the selection campaign (12 probability families × 8 decoders). Six pass the pre-registered rule (\*); the best, `short_gru25/dead45`, is selected. Removing the 1-s merge (`no_merge`) is the most damaging decoder change in every family.*

![Provider and holdout](image/fig05_provider_holdout.png)

*Figure 5. Final method vs the previous campaign's control. Left: development per provider. Right: holdout cohorts evaluated after the freeze (all matches had been seen in earlier campaigns; the ball-drop cohort removes 35% of visible-ball samples from the same six SkillCorner matches).*

| Development provider | Control (wave4) | Final (wave5) | Δ |
|---|---:|---:|---:|
| IDSSE | 0.584 | 0.612 | +0.028 |
| SkillCorner | 0.709 | 0.709 | +0.000 |
| Metrica | 0.677 | 0.689 | +0.012 |
| **Equal mean** | **0.656** | **0.670** | **+0.014** |

### 5.3 Where the method succeeds and fails

![Recall by duration](image/fig06_duration_recall.png)

*Figure 6. Recall of GT possessions by duration on the holdout cohorts (bars: IoU 0.5; red ticks: IoU 0.75). The short-possession loss raises recall of spells < 4 s in every cohort, but short recall stays below 25% at IoU 0.75.*

| Ball-drop stress (pooled, IoU 0.75) | Control | Final |
|---|---:|---:|
| Predicted possessions | 1,613 | 1,713 |
| Matched possessions | 744 | 751 |
| Precision | 46.13% | 43.84% |
| Recall of GT ≥ 15 s | 71.98% | 68.64% |

When more ball observations are removed, the final method creates more spurious possessions and loses some long ones. This is the main open problem.

![Output anatomy](image/fig07_output_anatomy.png)

*Figure 7. Submitted file. Left: possession durations predicted over the full match (302 possessions, median 6.7 s) vs the public GT. Centre: label shares. Right: frame confusion on the public window. Team errors are rare (4.9% of H frames, 1.2% of A frames), while 28.5% of dead-ball frames are labelled live: stoppages are detected late or shortened. The 44.1% dead share predicted over the full match cannot be checked without the private GT.*

### 5.4 From-scratch reproduction

The submitted file was produced by [`experiments/E06_wave5/E06_wave5.ipynb`](experiments/E06_wave5/E06_wave5.ipynb) with `RETRAIN = True` (run on an NVIDIA L40S). The notebook downloads the public training data, rebuilds the training packs and features, trains both networks, rebuilds the target inputs from the raw match bundle and decodes, using the campaigns' own code through [`train_from_scratch.py`](experiments/E06_wave5/train_from_scratch.py). Two exactness checks pass inside the run: a per-seed re-implementation matches `models.predict` (max abs. difference 1.8e-7), and the step-by-step decoder rebuild equals `decoder.decode`.

| | Submitted run |
|---|---|
| CSV SHA-256 | `850e6d25…` |
| Training | BiGRU 260 s on CPU (1 thread), PP-TCN 79 s on GPU (3 seeds × 1,500 updates) |
| Training possession rate ρ / dead share | 3.643 / 0.461 (provider-balanced, 19 matches) |
| Full-match possessions | 302 (median 6.7 s) |
| Public spellF1 / macroF1 | 0.772414 / 0.901002 |

Retraining on other hardware or library versions can change the weights slightly, so a rerun is expected to agree closely with the submitted file but not byte for byte. The external development and holdout scores in [§5.2](#52-selection-on-external-data) come from the selection campaign's own training run. Record: [`wave5_retrain_verification.json`](experiments/E06_wave5/results/wave5_retrain_verification.json).

### 5.5 Explainability

All figures in this section are outputs of the same notebook run, computed on the submitted model.

![Anatomy of a decision](image/fig10_decision_anatomy.png)

*Figure 10. One decision through every stage on the public window: (A) dead-ball probability of each component and the resulting dead mask; (B) P(home | live) of each component; (C) the decoder's team evidence ℓ_H − ℓ_A, with the samples where the proximity gate fired; (D) labels from Viterbi, after the 1-s merge, and the GT.*

The two components agree on dead/live for 92.0% of samples and on the team (argmax) for 85.3%; fusion mostly smooths the TCN with the BiGRU (panels A–B). The proximity gate fires on only 4.3% of samples (7.0% in the public window), so the decoder relies on the network's probabilities almost everywhere. The calibrated switch cost hits the smallest grid value (λ = 0.05) and still decodes 302 possessions against the 386 implied by the training rate: the possession count is limited by the dead mask and the merge, not by λ.

**Stage ablation** (one stage changed at a time; public columns are diagnostics on 17 possessions):

![Stage ablation](image/fig15_stage_ablation.png)

*Figure 15. Stage ablation. (A) Full-match possessions per variant; the dashed line is the count implied by the training possession rate (386). (B) Public-window spellF1 (diagnostic).*

| Variant | Full-match possessions | Median (s) | Share < 4 s | D share | Public spellF1 | Public macroF1 |
|---|---:|---:|---:|---:|---:|---:|
| **Final (dead45)** | **302** | **6.7** | **0.374** | **0.441** | **0.772** | **0.901** |
| Decoder `wave4` / `postmerge` / `live_reset` | 299 | 6.6 | 0.358–0.361 | 0.434 | 0.772 | 0.896 |
| Decoder `dead1` | 305 | 6.2 | 0.377 | 0.435 | 0.772 | 0.896 |
| Decoder `dead55` | 300 | 6.8 | 0.350 | 0.427 | 0.772 | 0.889 |
| No 1-s merge | 352 | 5.0 | 0.435 | 0.434 | 0.761 | 0.895 |
| Proximity gate off | 297 | 6.8 | 0.364 | 0.441 | 0.772 | 0.901 |
| BiGRU weight 0 (TCN only) | 287 | 7.4 | 0.352 | 0.444 | 0.786 | 0.921 |
| BiGRU weight 0.5 | 330 | 6.0 | 0.385 | 0.433 | 0.700 | 0.863 |
| BiGRU weight 1 (BiGRU only) | 388 | 5.3 | 0.415 | 0.410 | 0.566 | 0.707 |
| λ fixed at ½× / 2× | 306 / 284 | 6.6 / 7.4 | 0.376 / 0.349 | 0.441 | 0.772 | 0.901 |

Removing the merge fragments the output (352 possessions, spellF1 drops). Giving the BiGRU more weight over-segments sharply. The TCN alone scores higher on the public window (0.786); with 17 possessions this is within the noise of one possession (≈ 0.06) and was not used to change the pre-registered method.

![Integrated Gradients](image/fig11_integrated_gradients.png)

*Figure 11. Integrated Gradients: share of total |attribution| per input for the dead-ball logit (A) and the team logit (B), mean ± sd over the three seeds (16 crops × 512 samples, 64 steps; completeness error median 0.9%).*

| Input block | Dead logit | Team logit |
|---|---:|---:|
| Team-shape channels (23) | 41.6% | 49.6% |
| Player set (48 × 10) | 39.7% | 35.9% |
| Ball channels (5) | 16.7% | 12.6% |
| Validity flag | 2.0% | 2.0% |

The single most important input for both heads is the players' **attacking velocity** (x); the dead-ball head additionally relies on **ball speed**, nearest-opponent distance and player speed (play stops when motion stops). The ball contributes only 13–17%, which is why the method keeps working when the ball is missing. The ranking is stable across seeds (rank correlation 0.74 dead, 0.77 team).

![Temporal receptive field](image/fig12_temporal_receptive_field.png)

*Figure 12. Effective temporal receptive field of the PP-TCN: share of the input gradient by time offset from the decided sample.*

Although the receptive field spans ±12.8 s, 50% of the gradient mass lies within ±2.0–2.2 s and 90% within ±7.0 s, symmetric in past and future. The network decides locally; long-range consistency comes from the BiGRU and the decoder.

![Player attention](image/fig13_player_attention.png)

*Figure 13. Attention of the player-set encoder by the player's distance rank to the ball (A) and the weight on the nearest player relative to uniform (B).*

The attention pooling is **not** a "nearest player" detector: with the ball visible, attention increases with distance and the nearest player receives 0.83× the uniform weight; with the ball missing it is uniform (1.01×). Attention is split evenly between the two teams (50%). Proximity information reaches the model through the per-player ball and opponent distances (Figure 11) and the max-pooling path.

**Feature-group occlusion** (group set to its baseline over the whole match, full pipeline rerun):

| Occluded group | Labels changed | Possessions | Public spellF1 |
|---|---:|---:|---:|
| Team-shape channels | 13.0% | 395 | 0.484 |
| Player set | 7.5% | 238 | 0.600 |
| Ball channels | 4.0% | 276 | 0.787 |
| Validity flag | 0.8% | 305 | 0.772 |

![Feature-group occlusion](image/fig16_occlusion.png)

*Figure 16. Share of samples whose final label changes when one input group is occluded.*

| Occluded group | Mean \|Δ P(dead)\| | Mean \|Δ P(home \| live)\| |
|---|---:|---:|
| Team-shape channels | 0.095 | 0.132 |
| Player set | 0.058 | 0.045 |
| Ball channels | 0.035 | 0.024 |
| Validity flag | 0.008 | 0.004 |

Team shape is load-bearing (13% of labels change, public spellF1 collapses to 0.484), the player set is second; removing the ball channels changes only 4% of labels. Without the ball channels the output has fewer possessions (276 instead of 302) and the public spellF1 is unchanged within noise (0.787); the ball mainly informs the dead-ball head (mean |Δ P(dead)| 0.035 vs 0.024 for the team head). Without the player set possessions merge (238); without team shape they fragment (395).

![Seed disagreement](image/fig14_seed_disagreement.png)

*Figure 14. Per-seed P(home) on the public window (A) and the standard deviation across the three seeds (B).*

The seeds agree on the team almost everywhere (sd of P(home) 0.033 near decoded changes vs 0.032 elsewhere) but disagree on **stoppages**: the sd of P(dead) is 0.038 within ±2 s of a decoded change vs 0.009 elsewhere (4.3×). Model uncertainty is concentrated on when play stops and restarts.

**Error anatomy on the public window.** 11 of 17 GT possessions are recovered at IoU 0.5; recovered possessions last 17.8 s on average, missed ones 7.5 s. The misses cluster between minutes 9.9 and 11.6: three short Barcelona touches (0.8, 1.3 and 3.5 s) inside Athletic possessions and the Athletic spells they split. The ball is visible throughout these misses and the model's P(home) stays at 0.09–0.46, so the failure is brief contested exchanges, not ball loss.

![Error anatomy](image/fig17_error_anatomy.png)

*Figure 17. (A) Best IoU of each public GT possession against the prediction, by duration (red: home, charcoal: away; size: ball visibility); dashed line: IoU 0.5. (B) Frame confusion on the public window.*

| Missed Barcelona touch | Start (min) | Duration (s) | Ball visible | Mean P(home) | Seed sd |
|---|---:|---:|---:|---:|---:|
| 1 | 10.23 | 0.76 | 100% | 0.161 | 0.051 |
| 2 | 10.51 | 1.32 | 100% | 0.091 | 0.013 |
| 3 | 11.53 | 3.48 | 100% | 0.455 | 0.077 |

These misses are **systematic, not random**: on two of the three touches all seeds agree that Athletic has the ball (seed sd ≤ 0.05), so more seeds or a larger ensemble would not recover them. Only the 3.5-s touch is borderline (P(home) 0.46, the highest seed spread of any GT possession). Long recovered possessions (> 10 s) have P(home) > 0.95 or < 0.07 with seed sd ≤ 0.02. Every possession shorter than 2 s is missed, which matches the 1-s merge and the network's ±2 s effective context: a sub-second touch cannot outweigh the surrounding evidence.

## 6. Research history

Fourteen experiments were run in sequence; each campaign's control is the previous campaign's result. Each folder has a record card (`CARD.md`) and a README with test steps. Index: [experiments/REGISTRY.md](experiments/REGISTRY.md).

| ID | Experiment | Question | Primary result | Verdict |
|---|---|---|---|---|
| [E00](experiments/E00_eda/) | EDA | What does the tracking look like? | ball missing on 40.7% of graded frames | shaped the design |
| [E01](experiments/E01_baseline/) | Official baseline | How does nearest player + vote behave? | public 0.326, full match 0.2433 | reference floor |
| [E02](experiments/E02_wave1/) | Wave1 | Rules vs learned models on Metrica | hysteresis, dev 0.461 | superseded |
| [E03](experiments/E03_wave2/) | Wave2 | Gradient boosting + Viterbi | dev 0.517 | superseded |
| [E04](experiments/E04_wave3/) | Wave3 | Set+BiGRU with control features | dev 0.615 | superseded (BiGRU reused) |
| [E05](experiments/E05_wave4/) | Wave4 | PP-TCN across three providers | dev 0.656 | control of E06 |
| [E06](experiments/E06_wave5/) | **Wave5** | **Short-possession loss** | **dev 0.670 (+0.014)** | **final method** |
| [E07](experiments/E07_wave6/) | Wave6 | Fusion/decoder only | dev 0.675 (+0.0047 < 0.005) | not promoted |
| [E08](experiments/E08_fair_idsse/) | Fair benchmark | Families on one split and decoder | short-loss PP-TCN 0.596 | benchmark |
| [E09](experiments/E09_cross_branch/) | Cross-branch | All branches' cues combined | 0.609 (+0.013) | research primary passed |
| [E10](experiments/E10_temporal_methods/) | Temporal decoders | Duration HSMM, consensus | 0.606 (−0.010) | rejected |
| [E11](experiments/E11_spell_training/) | Spell training | Event sampling, spell loss | 0.619 (+0.003 < 0.005) | not promoted |
| [E12](experiments/E12_event_learning/) | Event learning | Six training recipes | 0.619 (−0.0004) | rejected |
| [E13](experiments/E13_provider_generalization/) | Provider transfer | Regularisers for unseen providers | 0.530 (−0.010) | rejected |

![Decision forest](image/fig08_decision_forest.png)

*Figure 8. Every campaign's pre-registered primary against its own control, with 95% paired match-bootstrap intervals where recorded and the pre-registered minimum gain (black tick). Only E06 (the final method) and E09 (research-only) passed; E07 missed its threshold by 0.0003 and was not promoted even though it scores higher on the public window (0.786). Cohorts differ between rows, so compare each row only with its own threshold.*

![Fair benchmark](image/fig09_fair_benchmark.png)

*Figure 9. E08: the team's model families retrained on the same 7 IDSSE leave-one-match-out folds and decoded with the same decoder. Grey dots are individual held-out matches. The short-loss PP-TCN leads on the mean, but its interval against SetSeq still crosses zero.*

## 7. Limitations and threats to validity

- **Adaptive evaluation.** All external holdout matches and the public window had been seen by earlier campaigns. Results are adaptive estimates, not a fresh blind test. Bootstrap intervals are not corrected for multiple comparisons.
- **Proxy labels.** SkillCorner (phases of play) and Metrica (events) differ from the corrected-tracking definition of the target GT.
- **Short possessions and ball loss.** Short-spell recall is below 25% at IoU 0.75. Under heavy ball loss, precision and long-spell recall drop ([§5.3](#53-where-the-method-succeeds-and-fails)).
- **Offline method.** Centred convolutions and the bidirectional GRU use future context. The method is not real-time.
- **Reproduction.** The submitted CSV is rebuilt end-to-end (public training data → training → raw match bundle → CSV) by one notebook ([§5.4](#54-from-scratch-reproduction)). Retraining on other hardware or library versions can change the weights slightly, so a rerun is expected to agree closely but not byte for byte. The external development and holdout scores come from the selection campaign's own training run.
- **Leakage audit.** No path from competition GT into training, inference or selection was found ([E13 exposure audit](experiments/E13_provider_generalization/EXPOSURE.md)). This is not a certificate of being leakage-free.
- **Licence.** Metrica sample data has no explicit licence; it is used only for evaluation.

## 8. Reproducibility

**Without match data** (fresh clone):

```bash
pip install -r requirements.txt
python evaluation/validate_submission.py carried_by_admin.csv    # official format check
python -B submission/export.py --output carried_by_admin.csv     # re-export; must print SHA256 850e6d25…
python experiments/check_results.py                        # 72 checks of every experiment's recorded results
cp carried_by_admin.csv evaluation/results/wave5.csv && python evaluation/run_evaluation.py   # public diagnostic (§5.1)
python -B -m pytest -q experiments/E06_wave5                # unit tests of one experiment (needs its requirements.txt)
```

**With the match data:**

1. Enrol, download the bundle and unpack it into `data/` (or set `MTAH_DATA_DIR`), as described in [DATA.md](DATA.md).
2. Data analysis: [`experiments/E00_eda/E00_eda.ipynb`](experiments/E00_eda/E00_eda.ipynb). Baselines: `python starter/baseline.py` ([E01](experiments/E01_baseline/README.md)).
3. Train from scratch, rebuild and explain the final CSV: run [`experiments/E06_wave5/E06_wave5.ipynb`](experiments/E06_wave5/E06_wave5.ipynb) (`RETRAIN = True`; a GPU is recommended for the PP-TCN). It downloads IDSSE and SkillCorner, trains both networks, builds the target inputs from `data/`, writes the CSV, compares it with the files above, and runs the explainability analyses of [§5.5](#55-explainability). The two halves can also run separately: `python experiments/E06_wave5/train_from_scratch.py public` on a GPU machine (no challenge data) and `... target` on the licensed machine ([E06 README](experiments/E06_wave5/README.md)).
4. Technical report: [docs/report/README.md](docs/report/README.md).

`requirements.txt` installs everything every experiment script and test needs (Python 3.10, campaign pins); `requirements-starter.txt` adds Jupyter, kloppy and the Typst compiler for the notebooks and the report. Each experiment also has its own `experiments/<EXX>/requirements.txt`, which resolves to the same pins.

## 9. Repository layout

```
carried_by_admin.csv        final possession file (method retrained from scratch)
carried_by_admin.pdf        technical document
image/                      README figures and the script that draws them (make_figures.py)
submission/                 export tool: export.py, method.json, artifacts/, verification records
web/                        possession review dashboard
docs/
  FINAL_METHOD.md           locked description of the final method (facts as data in final_method.json)
  BRANCHES.md               what each team branch contains; how to read historical numbers
  report/                   Typst source of the technical report, figure script, bibliography
experiments/
  REGISTRY.md               every experiment in run order
  E00_eda/ … E13_*/         one folder per experiment: code, CARD.md, README.md (how to test), results/
  check_results.py          recomputes every experiment's headline numbers from its results/*.json
  CAMPAIGNS.md              inputs and environment shared by the scripted campaigns E02–E13
  common.py, _template/     shared helpers and the template for a new experiment
evaluation/                 official scorer, validator, public 5-minute GT, baseline CSVs
starter/                    organisers' loaders, baseline and tutorial notebooks
games.py                    per-match constants, asset paths and frame alignment
data/                       unpack the match bundle here (gitignored)
```

Match data, caches, model weights and per-frame predictions other than the submitted CSV are not in the repository; the match data is not redistributable.

## 10. Challenge documents, licence and citation

Organisers' documents defining the task and rules: [CHALLENGE.md](CHALLENGE.md) · [DATA.md](DATA.md) · [EVALUATION.md](EVALUATION.md) · [SUBMISSION.md](SUBMISSION.md) · [RULES.md](RULES.md) · [FAQ.md](FAQ.md) · [OVERVIEW.md](OVERVIEW.md).

The challenge code and documentation are released by the organisers under the MIT Licence ([LICENSE](LICENSE)). **The match data is not covered by it**: it is licensed for this competition only. External data: IDSSE (CC BY 4.0), SkillCorner open data (MIT), Metrica sample data (evaluation only).

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}

@article{bassek2025,
  author  = {Bassek, Manuel and Rein, Robert and Weber, Hendrik and Memmert, Daniel},
  title   = {An integrated dataset of spatiotemporal and event data in elite soccer},
  journal = {Scientific Data}, volume = {12}, number = {1}, pages = {195}, year = {2025},
  doi     = {10.1038/s41597-025-04505-y}
}
```

Figures were produced with Matplotlib using colour-blind-safe Okabe–Ito colours, with the help of the Scientific Agent Skills library (Kassis et al., 2026, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)).

## 11. Web review application

English match-review application for the saved result, implemented from the approved FigCraft design. Live application: <https://possession-match-review.vercel.app>.

- Routes: `/` for possession overview, measured shares and reference/model comparison; `/replay` for local video with synchronized 2D method replay, frame stepping and aligned timelines; `/evidence` for published scores, sources and evaluation limitations.
- Run locally from `web/` with Node.js 24 and npm: `npm ci`, then `npm run dev`, and open `http://127.0.0.1:3000`. Production server: `npm test`, `npm run build`, `npm start`.
- Video input is selected by the user. The five-minute reference clip is the 300-second tactical clip starting at original recording time `07:27.40` (frame 11185). The original tactical recording is limited to `07:27.40` through the last frame before `12:27.40`. MP4 with H.264 video is recommended.
- All views use source frames `[11185, 18685)`: 7,500 frames at 25 fps. The bundled model segments match the same window of `submission/artifacts/farras_wave5.csv`.
- Public reference labels are fetched in the browser from a pinned public repository revision; a local reference file can be opened if that source is unavailable. Files must have continuous absolute `frame,label` rows for the entire match or the exact public window.
- `/replay` can load `method-replay.private.json` for synchronized observed tracking positions and saved model outputs. Neither that file nor match media is deployed.
- Deployment is CLI-managed from `web/`: `npx vercel deploy --prod --yes --scope eycoos-projects`.

Full details, including monorepo layout, exporter usage and verification, are in [web/README.md](web/README.md).
