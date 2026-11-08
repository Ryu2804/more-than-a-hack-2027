# exp_2 pre-registration — transfer-robust SetSeq

Written 2026-11-08, before any exp_2 model fit. Its SHA-256 is stored in `results/protocol.json`
together with every constant below as read from the code; each notebook re-checks both and
stops on any difference.

## 0. What was looked at before freezing (not blind)

* All exp_1 results (`exp_1/REPORT.md`): exp_1 SetSeq 0.656 mean spellF1, under-segmentation
  (−4 % to −34 %), boundary-aware decoding never selected.
* Label-free aggregates of the 7 round-0 degraded IDSSE matches, used to calibrate the stress view:
  ball visible 59.0 %, visible ball off the pitch on 26.4 % of visible frames (BAR-ATH: 4.2 %);
  keeping the first 37 frames of each off-pitch run gives 4.2 % pooled. Label-based descriptives
  (not used for any choice): IDSSE off-pitch share of visible *dead* frames 59 %, dead share 45 %,
  smoothed collective speed < 1 m/s marks 22 % of frames, 98 % of them dead.
* Run-length statistics of the 5 fps labels (team spells median 19 steps, max 465; dead spells
  median 98, max 1033) to set `D_MAX`.
* Decoder smoke test on **one** cached exp_1 outer prediction (J03WMX, original view): the
  un-centred duration log-pmf added ≈3–5 nats per segment and *reduced* the count (354 → 297 as w
  grew), so the duration term is entropy-centred (§4). No grid value was chosen from this test.
* Licences of extra data sources (`DATA_SOURCES.md`).

## 1. Hypotheses

* **H1 (primary)** — training SetSeq with target-like observation augmentation improves spellF1 on
  the `target_like` view of the held-out matches and keeps original-view spellF1 within 0.02 of exp_1.
* **H2 (secondary, report only)** — an explicit-duration decoder with label-estimated duration priors
  brings every held-out match within ±30 % count error without losing more than 0.01 spellF1 vs the
  constant decoder on the same predictions.
* **H3 (secondary, report only)** — averaging 3 training seeds' probabilities changes spellF1 by more
  than the seed spread.

## 2. Data and views (same 7 IDSSE matches, same frozen judge as exp_1)

Held-out matches are always exp_0's frozen corruption, round 0 (exp_1 snapshot, hash-checked).

* **`original`** — the round-0 degraded match, exactly as in exp_1.
* **`target_like`** (primary for H1–H3) — label-free, deterministic, applied on top of round 0 to the
  held-out match for *every* arm, including `exp1_ref`, ryu's zero-shot pipeline and the rule arms:
  1. off-pitch ball: in each maximal run of graded frames whose visible ball lies strictly outside
     [0, L] × [0, W], keep the first `K_OUT = 37` frames (1.48 s) and hide the rest;
  2. slow play: collective speed = mean player speed over present slots (no players → 2.0 m/s, as the
     frozen degrader), moving average over 51 frames inside each period; hide the ball on every frame
     of runs with smoothed speed < 1.0 m/s lasting ≥ 125 frames (5 s). Threshold fixed a priori from
     the team EDA (collective speed 0.6–1.5 m/s before restarts);
  3. players, teams, halves and labels unchanged.
  Pre-freeze label-free numbers: ball visible 45.3 % after step 1, 39.5 % after step 2 (BAR-ATH
  59.3 %). The view is *harsher* than BAR-ATH on ball coverage; the frozen evaluation corruption
  cannot be reduced, so this is reported, not corrected.
* **`target_aug`** (training only; may use the training match's labels). Applied to corruption rounds
  1 and 2 of each training match; round 0 stays original, so a model sees one original and two
  augmented copies (same data volume as exp_1):
  1. off-pitch ball: per off-pitch run keep the first k frames, k ~ U{0, …, 74} (mean 37 = `K_OUT`);
  2. dead spells (label D) lasting ≥ 125 frames: with probability 0.5, hide the ball from 25 frames
     after the spell starts to 25 frames before it ends.
  RNG seed `20261108 + 1000·round + match index + 7 000 000`. The rate 0.5 is fixed a priori: after
  step 1 coverage (≈45 %) is already below BAR-ATH's 59.3 %, so no hiding rate can match it.
  The training transform deliberately differs from the label-free `target_like` rule.

## 3. Arms

| arm | predictions | decoder |
|---|---|---|
| `exp1_ref` | exp_1 SetSeq, `TrainConfig()` (seed 20261108), rounds 0–2 original (exp_1 token cache). Re-fit because exp_1 saved no weights; reproduction of exp_1's cached original-view predictions is reported (max abs difference). | exp_1 main (`setseq_bd`): γ ∈ {0, 0.5, 1, 2}, λ_team, λ_dead ∈ {0.5, 1, 2, 4, 8, 16} |
| `exp2_aug` | same network and config, rounds 0 original + 1–2 `target_aug` | as `exp1_ref` |
| `exp2_aug_const` | `exp2_aug` | γ = 0 only (constant decoder; H2 reference) |
| `exp2_aug_hsmm` | `exp2_aug` | explicit-duration decoder (§4) |
| `exp2_aug_s1`, `exp2_aug_s2` | as `exp2_aug`, seeds 20261109 / 20261110 | as `exp1_ref` (H3 spread only) |
| `exp2_aug_ens3` | per-step mean of p(H,A,D) and p(boundary) of the three seeds | as `exp1_ref` |
| `exp2_aug_multi` | **not run** — see `DATA_SOURCES.md` (decided before any fit) | – |
| reference: `baseline`, `rasya_c06`, `rasya_c08`, `ryu_final` | no fitting; rasya/ryu constants as frozen in exp_1 | own |

Diagnostics (no decision): argmax (no decoder) for `exp1_ref` and `exp2_aug`. exp_0's LightGBM is not
an exp_2 arm (exp_1 reports it on the original view). Fit budget: 4 arms × 28 = 112 proxy fits +
3 full-data fits for the BAR-ATH aggregates (+1 rebuild fit if a candidate is adopted) ≤ 116.

## 4. H2 decoder

Segment of state s ∈ {H, A, D} covering d steps (5 fps) scores
`Σ log p_T + w · (log P_s(d) + entropy(P_s)) − λ(previous → s)`, λ_team for H↔A, λ_dead for changes
touching D. The entropy term centres the duration score (zero mean under P_s). A segment may be followed
by one of the same state only if it has length `D_MAX = 1100` steps (220 s; continuation).
P_team (H, A pooled) and P_dead are discretised log-normals (d = 1…D_MAX) with μ, σ = mean, SD of log
run lengths of the 5 fps labels of the matches the decoded model was **trained** on (pair model: 5,
outer: 6, BAR-ATH: 7); unlabelled steps break runs. Grid: w ∈ {0.25, 0.5, 1}, λ_team, λ_dead ∈
{0, 0.5, 1, 2}; temperature by NLL as in exp_1. Boundary head unused.

## 5. Protocol (unchanged from exp_1)

Nested leave-one-match-out per view: 21 pair models give inner validation predictions, 7 outer models
the held-out predictions; every fit predicts both views of its held-out matches with the same weights.
Temperature and decoder settings are chosen on the six inner validation matches of the same view with
exp_0's objective (mean spellF1 − 0.15 × mean excess |count error| above 0.30); the held-out match is
scored once with the official `possession_metrics.py`. The BAR-ATH decoder config is selected on the 7
outer out-of-fold predictions of the `target_like` view. Paired match bootstrap, 1000 draws, seed
20261108, 95 % percentile interval.

## 6. Decision rules (fixed now)

* **H1a** — Δ = spellF1(`exp2_aug`) − spellF1(`exp1_ref`) on `target_like`: 95 % CI lower bound > 0.
* **H1b** — same Δ on `original`: mean ≥ −0.02.
* **H1 supported** iff H1a and H1b.
* **H2 supported** iff, on `target_like`, every match of `exp2_aug_hsmm` has |count error| ≤ 0.30
  **and** Δ(`exp2_aug_hsmm` − `exp2_aug_const`) has 95 % CI lower bound > −0.01.
* **H3 supported** iff, on `target_like`, |mean Δ(`exp2_aug_ens3` − `exp2_aug`)| exceeds the seed
  spread (max − min of mean spellF1 over `exp2_aug`, `exp2_aug_s1`, `exp2_aug_s2`) **and** the 95 % CI
  of that Δ excludes 0.
* H2 and H3 are also reported on `original`; that never changes a decision.
* **Adoption (BAR-ATH candidate)** iff H1 supported. The candidate is always `exp2_aug` (seed
  20261108, exp_1 main decoder, final config from §5); H2/H3 never change it. If adopted: write
  `predictions/exp_2_candidate.csv`, pass the official validator, rebuild byte-identically from an
  independent fit, aggregates only. **No public-window check** (spent by exp_1).
* Reported for every arm and view: mean/min spellF1, macroF1, mean/max |count error|, exp_0 quality
  gates vs the same view's baseline (including the ±30 % count gate), paired Δ vs `exp1_ref` with
  95 % CI.

## 7. BAR-ATH plausibility (label-free, report only)

Predicted dead share, possessions per minute and mean possession duration for exp_1's shipped CSV (from
`exp_1/results/target_summary.json`), `exp2_aug`, `exp2_aug_hsmm` and `exp2_aug_ens3` (full-data fits,
final configs), against the published prior dead share 0.407. Never used for any choice.

## 8. Hygiene

BAR-ATH is read only in notebook 05's kernel, which prints aggregates only; its executed notebook is
not opened, results are read from `results/*.json`. No public ground truth, no hand labels, nothing
under `data/` written, no frame-level BAR-ATH export besides the gitignored candidate CSV.
