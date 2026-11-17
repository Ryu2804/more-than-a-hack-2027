# Public exposure audit and limits of generalisation

## What was verified in the old code

The competition's RULES.md 52-84 forbids training/tuning on human labels of the provided match; the public evaluation window is only for pipeline correctness. EVALUATION.md 125-142 explains that 17 possessions are not enough to choose a model from the public ranking.

The inspected selection logic uses external-development metrics and freezes the choice before public diagnostic scoring:

| Campaign | External selection | Public access after the freeze |
|---|---|---|
| wave3 | run.py 241-273 | run.py 417-438, GT load 423 |
| wave4 | run.py 172-210 | run.py 363-383, GT load 370 |
| wave5 | run.py 224-240 | run.py 387-409, GT load 396 |
| wave6 | run.py 125-139 | run.py 212-233, GT load 222 |

No public metric flows back into those selection branches. Wave6/RESULTS.md line 7 keeps wave5 when the candidate fails the external minimum gain; the 0.786207 public diagnostic does not reverse the decision. This is evidence of the decision order in the inspected code, **not a guarantee that already-known public scores never influenced later hypotheses**. Wave5/PLAN.md 5-7 and wave6/PLAN.md 30-32 themselves mention the exposure and the adaptive research.

The recent fair_idsse/cross_branch/spell_training/event_learning runners score externally and record `new_public_evaluation=False`. That flag applies to those campaigns; it does not erase the historical public access. Using the previous best exploratory result as the new control is also a form of adaptive research, already reported.

(Line numbers refer to the files at the time of the audit, in `experiments/exp_farras/<campaign>/`; those folders are now `experiments/E04_wave3/` … `E07_wave6/`.)

## All 27 external matches have been exposed

Wave4/campaign.py 15-24 defines 7 IDSSE and 20 SkillCorner matches. Wave3/PLAN.md line 15 acknowledges that the seven IDSSE matches were used by Jeremy. Wave4/PLAN.md 9-15 records that the shipped weights were trained on the 27 matches; the historical SkillCorner holdouts were also used by Ryu. Wave5/RESULTS.md 60-64 acknowledges that holdout exposure. Renaming a split does not make it blind.

The [current SkillCorner match tree](https://api.github.com/repos/SkillCorner/opendata/git/trees/0c809ab9eabe28ebcd0b28a6916617cde88f852c) contains the same 20 IDs as the local SC_IDS. There is no fresh cohort in this inventory. The upstream README mentions 10 matches, but the tree and matches.json contain 20; the cohort is determined from the actual inventory, not from the README number. No new tracking was downloaded.

## Direct leakage versus a label-access barrier

The inspected feature path separates possession labels from observations:

- wave3/reference/jeremy_data.py 125-162 separates metadata/team mapping, tracking and labels; corruption 198-276 receives observations, not labels.
- wave3/prepare_idsse.py 293-317 builds X16 before extracting labels.
- wave4/prepare_skillcorner.py 457-497 separates phase labels from interpolation/distances/X16.
- wave1/data.py 76-113 builds X16 from positions, distances, visibility and motion/gaps.
- wave4/features.py 530-576 builds PP/pooled/ball features from observations; direction inference 121-161 uses player positions, not agreement with labels.
- The historical target carrier wave3/prepare_local.py 339-381, 394-418 uses labels None, later unknown −1.

No use of possession GT as a feature was found on these paths. However, the inherited loader wave4/campaign.py 63-80 **reads y/y25 before observations() drops them**. That does not prove labels affected inference, but it does not meet a literal claim that inference cannot read test labels. The new campaign uses a carrier without y/y25 and a runtime read guard against the original label-bearing packs/vault.

The inherited training helper wave4/campaign.py 76-78 also adds r0/r1/r2 for IDSSE. The new source-only protocol calls the r0 loader explicitly; that helper is not used.

## What may and may not be concluded after the new experiment

May be concluded, if every verification passes: fresh source-provider-only training; match/provider disjoint per direction; source-only normalisation/rate/weights; label-free inference; every prediction committed before any new metric; competition-public reads refused by a trusted-code hook; results on external datasets that were already exposed.

May not be concluded: freedom from overfitting, never having seen the public window, a fresh blind test, unbiased performance, a private/full-match improvement, or an OS-level confidentiality guarantee. Python audit hooks do not cover all native I/O, subprocess leakage or adversarial bypass. The new clean/gap conditions are paired variants of 27 matches, not 54 independent matches. Provider transfer mixes domain shift, label/proxy definitions and tracking quality; a score difference between providers is not evidence of overfitting on its own.

Main decision and new method: [PLAN.md](PLAN.md). Execution and verification results: [RESULTS.md](RESULTS.md). The public competition data was not opened by this audit; the evidence on the rules comes from documentation and code, not from the content of the GT CSV.
