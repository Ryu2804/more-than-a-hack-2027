# E11: Spell-targeted training

| | |
|---|---|
| **Folder** | [`experiments/E11_spell_training/`](./) |
| **Builds on** | E10 (control 0.615458) |
| **Status** | done |
| **Verdict** | not promoted (gain 0.003484 < 0.005) |

## Question
Does changing what the network learns about short spells (loss, crop sampling, or temporal kernel) help more than further decoder smoothing?

## Hypothesis
Primary: event-centred crop sampling (`event_replace25/dead45`) improves spellF1 over the control.

## Design
- **Three new families**, one factor each: spell-balanced loss (inverse-sqrt duration weights), event TCN (50% of crops centred on complete spells < 4 s), dual-scale TCN (shared kernel at dilation 1 and original dilation).
- **Split and budget:** 7 IDSSE LOSO folds, 63 fresh seed fits, 20 settings / 140 evaluations.

## Metric and decision rule
Gain ≥ 0.005, wins ≥ 5/7, paired-bootstrap 95% lower bound > 0 (5,000 draws, seed 20261117).

## Result
Primary `event_replace25/dead45`: 0.618941, Δ +0.003484, CI [0.000670, 0.006504], 5/7.
Best exploratory `event_replace25/consensus`: 0.619563. Against a control with the same consensus decoder: Δ +0.002641, CI [−0.000511, 0.006012].
Standalone event TCN 0.608023 vs original short-loss TCN 0.595755.

## Verdict
**Not promoted.** Positive interval but gain below the 0.005 minimum. The exploratory winner becomes the strong control for E12.

## Threats to validity
Part of the gain against dead45 comes from the decoder, not the training change.

## Evidence
[`spell_training/PLAN.md`](PLAN.md), [`spell_training/RESULTS.md`](RESULTS.md), [`results/spell_training_*.json`](results/).
