# Temporal methods: frozen protocol

## Starting evidence

The preceding cross-branch campaign's best exploratory configuration was a fixed probability mixture, 75% short-loss PP-TCN plus 25% SetSeq, decoded with dead45. Mean official spellF1 on seven IDSSE LOSO matches: 0.6154575861548992. This previously selected configuration is now the control; the matches and earlier hypotheses have already been observed. This is another adaptive research campaign, not a fresh blind estimate or a competition submission.

Recomputed control diagnosis before this campaign: 3,061 ground-truth H/A spells versus 2,208 predicted; mean per-match relative count error -0.2738908051172179. At diagnostic IoU 0.75, pooled matched recall was 230/1,621 for spells under 4 seconds, 539/1,058 for 4 to under 15 seconds, and 301/382 for spells at least 15 seconds. IoU 0.75 is diagnostic only, not the competition's primary metric.

## Fixed inputs and evaluation

- Exactly seven original IDSSE matches, one held out and six other matches for each fold. No synthetic validation or pretrained weights containing the held-out match.
- Reuse the audited original fair benchmark's held-out short-loss TCN and SetSeq probabilities. Both models had three seeds, 500 updates per seed, identical training membership and budgets. No new neural fits in this campaign.
- All new decoders use the same fixed 75/25 joint mixture. Consensus additionally sees the two individual held-out predictions and the already audited raw-25fps contact features from cross_branch. All features are observation-only. No ground-truth orientation, D/X mask, or held-out label calibration.
- Duration priors, initial state fractions and transition frequencies are fitted only to the six training matches' raw labels. This is not meta-training on other folds' OOF predictions, whose base fits could have seen the outer test match.
- Preserve half boundaries, 5fps prediction grid, block-aligned 25fps expansion, original provider ground truth, unknown-label masking and every official H/A spell, including short spells.
- Primary metric: mean per-match official spellF1 across IoU 0.3/0.4/0.5/0.6/0.7. MacroF1 secondary. Equal decoding inputs, not equal operation counts or runtime.
- No new public or target evaluation, no target CSV, no automatic commit or push.

## Three new methods

### Explicit-duration HSMM

A joint three-state age-expanded Viterbi decoder replaces independent dead gating and two-state team decoding. Per-state duration hazards come from complete runs in six training matches. First and last runs of each contiguous known interval within each half are censored and excluded from the duration histogram. Unknown gaps do not create transitions. Initial state fractions use smoothed known-frame counts; off-diagonal transition frequencies receive one pseudocount each.

For ages 1 through 79, Beta(1,1) hazard is `(count(L == age) + 1) / (count(L >= age) + 2)`. Age 80 pools an unbounded geometric tail with hazard `(count(L >= 80) + 1) / (sum(max(L - 80 + 1, 0)) + 2)`. This is not a 16-second duration cap. Missing complete classes use the declared Beta(1,1) prior.

Joint log probabilities are emission potentials, not a claim of generative likelihood calibration. Staying adds weighted log survival; switching adds weighted log hazard and off-diagonal transition probability, minus switch bias. Initialize age 1 at each half and allow right-censored terminal runs without an exit factor. No minimum-length merging, rate targeting, or forced possession counts.

Six configurations: prior strength 0.25/0.5/1.0 crossed with switch bias -0.5/0.0.

### Consensus change-point penalties

Keep control dead45 gating, spatial evidence, live resets, one-second merge and full 18-point lambda calibration after merging. At frame t, compute signed changes in conditional home logits from both models. Agreement strength is `clip(min(abs(change_short), abs(change_setseq)) / 2, 0, 1)` if both signs agree, else zero. Reset at each half. Observed contact strength uses clipped contact acceleration channel 2 divided by 3, gated by a known contact-team sign in channel 3. Transition weight is `1 - alpha * clip(agreement + contact_beta * contact_strength, 0, 1)`.

Six configurations: alpha 0.25/0.5/0.75 crossed with contact beta 0.0/0.5. Full lambda-grid calibration uses only six-match training possession rate, not held-out labels.

### Ordered boundary relocation

Preserve the control's ordered state runs and run count in every half, including dead runs. Move each internal boundary by at most a fixed radius, with all new runs at least one sample long. Exact dynamic programming jointly optimizes the sum of joint log probabilities over the resulting runs minus `penalty * ((new_boundary - original_boundary) / radius)**2` per moved boundary. Half endpoints stay fixed. No independent greedy shifts, crossing boundaries or post-relocation merging.

Six configurations: radius 2/5/10 samples (0.4/1/2 seconds), crossed with penalty 0.0/0.05.

These are adaptations of explicit-duration sequence modeling and boundary-focused temporal segmentation, not reproductions of the cited papers: [explicit-duration models](https://www.sciencedirect.com/science/article/pii/S0004370209001416), [boundary-aware segmentation](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html).

## Decision fixed before evaluation

Nineteen configurations total: one control plus six for each new method. Evaluate all 133 configuration-match pairs. Primary is HSMM strength 0.5, switch bias 0.0. Promotion requires mean spellF1 gain at least 0.005 against the 0.6154575861548992 control, wins on at least five of seven matches, and paired match-bootstrap 95% lower bound above zero. Bootstrap: 5,000 draws, seed 20261116. Seven matches and repeated reuse limit confidence; intervals are not corrected for multiple comparisons or adaptive history.

Report all other configurations as exploratory. If primary fails, retain control instead of silently selecting the best new score. Also report raw best transparently. Source, this plan, predecessor provenance, cached probabilities and contact inputs are hashed before any new metric evaluation. A separate process refits every fold's duration priors and rebuilds every decoded output; all 133 array hashes and prior identities must match. Mathematical tests compare both exact dynamic programs against exhaustive tiny cases. Aggregate JSON reports are exportable; raw predictions, priors and caches stay under ignored out/.
