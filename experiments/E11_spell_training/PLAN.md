# Spell training: preregistered experiment

## Evidence and hypothesis

The preceding temporal_methods campaign tested19decoder settings. The promoted control remains fixed75% short-loss PP-TCN plus25% SetSeq, dead45, mean official spellF1 IDSSE0.6154575861548992. Exploratory consensus reached0.6169697856726231, but its interval crossed zero and gain fell below0.005. All explicit-duration HSMM and boundary-relocation variants lost to control. Control predicted2,208 of3,061H/A spells, with IoU0.75 diagnostic matches230/1,621for spells shorter than4seconds.

Hypothesis: improve information learned for short spells rather than add more decoding smoothness. This campaign trains three new neural families, separately changing loss, training-window selection, or temporal convolution. It does not retune earlier frozen campaigns.

## Fixed benchmark

Seven original IDSSE matches, seven LOSO folds, each trained on exactly six other unique matches. Original provider labels, unknown gaps masked, half boundaries preserved, same r0corruption and prepared features. No model that trained on heldout match, no GTorientation or oracle D/Xmask. No new data downloads or corruptions.

All three families use original PPencoder,29global channels and per-player48x10features with identical masks and two views. Width64, six residual blocks, Conv1d kernel5, dilations(1,2,4,8,16,1), GroupNorm8, GELU, dropout0.15. Three seeds(0,1,2),500optimizer updates/seed, batch16, crop384, AdamWlr0.002/weight_decay0.01, OneCycleLR, original ballmask_full0.3/noise0.5. Fresh random initialization; never load predecessor weights. Total63newseedfits/31,500updates. Equal update/input/parameter budgets, not equal compute: dual-scale adds convolution evaluations without adding parameters. Inference uses complete halves, two views and three-seed averaging, as preceding PP-TCN.

Baseline's original short-loss TCN and SetSeq probabilities are reused from authenticated six-match-exclusion OOF jobs; training memberships, manifest and probability bytes must match. No SetSeq retraining or new recurrent GPU model. Model caches and old source remain immutable.

## Three single-factor methods

### spell_balanced_tcn

Keep original architecture and uniform source/match/half-window sampling. Full-training-match H/A runs receive weight `clip(sqrt(20 / run_length),0.25,4)`, where run_length uses5fps samples. Only complete H/A runs qualify: any run touching a half endpoint or unknown gap receives1. Dweight1, unknownweight0. Compute weights before cropping; partial long spells cannot become short by window slicing.

Dead BCE uses ordinary known-frame mean. Conditional team BCE uses weighted live-frame mean, denominator `sum(weight * live_mask)`. This changes team emphasis rather than globally scaling its loss. Baseline's old short recipe weights both dead/team losses by2on H/Aspells shorter than20samples and keeps unweighted denominators; the distinction is intentional.

### event_tcn

Keep original architecture and original short-loss objective exactly. Choose source/match/variant/window/view with original balanced sampler first. Independently with probability0.5, replace that selected match's window with one centered on a uniformly chosen complete H/Aspell shorter than20samples. Only spells strictly inside a usable half and not adjacent to unknown labels qualify. Random center jitter is an integer in[-96,96], then clamp window to its half. Selected spell stays entirely inside384sample crop. If selected match has no eligible event, retain original drawn window and report zero eligible events explicitly. Never oversample event-rich matches globally or cross a half boundary.

### dual_scale_tcn

Keep original short-loss objective and original sampling. In each temporal block whose original dilation d>1, average two Conv5outputs before GELU/dropout: one at dilation d and one at dilation1. Both use the same registered weight and bias, without creating new kernels or parameters. Dilation1blocks remain one convolution. PPencoder, normalization, residuals and two-head output stay unchanged. This tests local evidence alongside longer context without increasing parameter count; convolution compute increases.

## Evaluation fixed before fits

Control probability `C=.75*short+.25*SetSeq`. For each new family N, report:

- pure: N;
- base75: `.75*C+.25*N`;
- replace25: `.50*short+.25*SetSeq+.25*N`.

One control plus nine new probability families, each decoded by frozen dead45 and frozen temporal_methods consensus(alpha0.5,contact_beta0.5). Consensus weights use old short/SetSeq agreement and the already audited observed raw25fps contact features, never new labels. Thus20configurations x7matches=140official metric rows. No parameter scan after scores are opened.

Primary: `event_replace25/dead45`, compared against `control/dead45`. Promotion requires mean spellF1 gain>=0.005, wins on>=5/7matches and paired match-bootstrap95%lower>0. Bootstrap5,000draws, seed20261117, unit unique match. If primary fails, control remains selected; exploratory winner is reported without automatic fallback.

Main metric: official mean H/AspellF1 over IoU0.3/0.4/0.5/0.6/0.7, including every short spell. MacroF1 secondary. Same block-aligned5fps-to25fps expansion, unknown masking and GTfor all arms. Duration-stratified recall at IoU0.75 is diagnostic only.

## Verification and scope

Before full fits: semantic tests for complete-run weights/censoring, masked gradients, loss denominators/view symmetry, event enrichment and match balance, half-safe crops, exact dual-scale outputs/gradients, PPpermutation/masking, parameter equality; actual1update/seed smoke training on six real matches and heldout inference for all three families. Smoke is not a performance estimate.

After smoke, hash source/PLAN/inputs and predecessor provenance before full500update fits. Each job runs in an isolated process and stores actual training names, three seed update counts, parameter counts, weights and probability SHA256. Separate isolated inference replay loads every saved model ensemble and requires exact equality to its cached full heldout probabilities. Final independent rebuild reconstructs140decoded arrays with matching hashes; all21model-replay proofs must match. This is inference and decoder reproduction, not another full training refit.

All seven matches and hypotheses have prior exposure. This adaptive research campaign is not fresh blind validation; bootstrap intervals do not correct repeated campaigns or multiple comparisons. No public/private evaluation, target CSV, automatic commit or push. New files only under spell_training and aggregate results prefix, separate from teammates' experiments. Raw tracking, weights, predictions and per-frame loss weights remain ignored.
