# Pre-registration — exp_1 follow-up: centred boundary-aware decoding

Written 2026-11-08, before any follow-up decoding was run. Its SHA-256 is stored in
`results/followup_protocol.json` and checked by notebook 05 before evaluation.

## Why (motivated by exp_1 outer results, so this is not blind)

In exp_1 the boundary term `gamma * log(b_t)` could only add switch cost, nested selection chose
gamma = 0 in every fold, the model under-segments (worst match -34%), and lambda sat at the lowest
grid value. The same 7 matches are reused; the nested protocol and this fixed rule limit, but do not
remove, the optimism of a follow-up on seen data.

## Change (one component only)

Decoder switch cost from state i to j at step t:

    cost_ij(t) = max(0, lambda_ij - gamma * (log b_t - log b_mean))

b_t is the SetSeq boundary probability, b_mean its mean over the same match (label-free). Staying costs
0. Above-average boundary probability now *lowers* the switch cost (down to 0), below-average raises it.
gamma = 0 is exactly the constant-penalty decoder. Model predictions are the cached exp_1 SetSeq fits
(no retraining). Temperature fitted by NLL on inner validation, as before.

## Grids (wider lambda, fixed now)

* lambda_team, lambda_dead in {0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16}
* arm `setseq_const_wide`: gamma in {0}
* arm `setseq_cbd_wide`: gamma in {0, 0.5, 1, 2}
* selection objective, nesting, bootstrap: identical to exp_1 (`results/protocol.json`).

## Decision rules (fixed now)

1. **Centring helps** iff paired delta `setseq_cbd_wide - setseq_const_wide` has 95% CI lower bound > 0.
2. **Grid edge mattered** iff paired delta `setseq_const_wide - setseq_const` has 95% CI lower bound > 0.
3. **Adopt a follow-up BAR-ATH candidate** iff paired delta `setseq_cbd_wide - setseq_bd` (frozen exp_1
   main) has 95% CI lower bound > 0. If adopted: final config chosen on the 7 out-of-fold predictions,
   CSV written as `predictions/exp_1_followup_cbd.csv`, aggregates only, byte-identical rebuild
   required. **No public-window check** (it was single-use and is spent).
4. The exp_0 count gate (all matches within +-30%) is reported for both new arms; it does not change
   rules 1-3.
