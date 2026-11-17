# E04: Wave3, identity-free set+BiGRU with control features

| | |
|---|---|
| **Folder** | [`experiments/E04_wave3/`](./) |
| **Builds on** | E03; adapts Jeremy branch `0e35aae` |
| **Status** | done |
| **Verdict** | selected `hybrid_tree25/centered2`; superseded by E05 |

## Question
Does Jeremy's identity-free temporal context (player sets + BiGRU) complement Farras's control evidence under matched observations and folds?

## Hypothesis
A set encoder + bidirectional GRU, mixed with a control-feature tree, beats either alone; a boundary-aware Viterbi cost helps.

## Design
- **Data added:** IDSSE (Bassek et al. 2025, CC BY 4.0) with a label-free degrader, 3 corruption rounds per match.
- **Split:** development IDSSE J03WMX, J03WN1, J03WOH, J03WOY, J03WPY; campaign holdout J03WQQ, J03WR9; Metrica 1/2 as cross-provider development stress; Metrica 3 as secondary holdout (already seen in E03).
- **Size:** 11 probability families, 66 configurations, 18 neural fits, 18 IDSSE tree fits, 594 development evaluations.
- **Neural recipe:** seed 2027, window 300, batch 16, 8 epochs, width/hidden 32, AdamW lr 0.002.

## Metric and decision rule
Rank by mean development spellF1; tie-break macroF1. Freeze before holdout and public.

## Result
Selected `hybrid_tree25 + centered2`: 75% set+BiGRU (plus 7 control features) and 25% Metrica-trained tree; centered boundary transition cost (λ_team 2, λ_dead 2, γ 0.5). Mean development spellF1 0.615394, macroF1 0.873323.

Post-freeze public diagnostics: wave3 0.575610 (24 predicted possessions, over-segmented), Jeremy's own branch report 0.581818, Farras wave2 0.460606.

## Verdict
Frozen; its GRU component is reused, frozen, in E05 and E06.

## Threats to validity
Development score is the maximum of 66 configurations (selection bias). The IDSSE matches had been used by Jeremy. Not comparable to Jeremy's 0.656 proxy (different split, recipe, decoder).

## Evidence
[`wave3/PLAN.md`](PLAN.md), [`wave3/RESULTS.md`](RESULTS.md), [`JEREMY_REVIEW.md`](JEREMY_REVIEW.md), [`results/wave3_*.json`](results/).
