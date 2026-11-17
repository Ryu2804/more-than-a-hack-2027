# Audit of Jeremy's branch

Pinned source: [`0e35aae54e09ca744441681fb076f656bbc88b5a`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/0e35aae54e09ca744441681fb076f656bbc88b5a/experiments/exp_1). The numbers in this section come from the branch's artifacts; the original checkpoint was not reproduced locally.

## Reported results

- `results/public_sanity.json`: spellF1 0.581818, macroF1 0.725033, 16 predicted possessions versus 17 GT; bootstrap spellF1 interval [0.40, 0.751515].
- `REPORT.md`: 7-match IDSSE proxy, mean spellF1 about 0.656 and macroF1 about 0.918. On the same proxy, Ryu's zero-shot reference is about 0.487.
- `results/followup_decision.json`: the centered-boundary follow-up did not pass all three adoption rules; no follow-up submission was made.
- `results/target_summary.json`: decoder temperature 1.302867, lambda_team 0.5, lambda_dead 0.5, gamma 0; 437 predicted target possessions; valid format and reproduction hash recorded by Jeremy.

Compared with Farras's 0.56 diagnostic, Jeremy's public score is 0.021818 higher. The window has only 17 GT possessions and Jeremy's interval is wide. This does not prove a private/full-match advantage or a significant difference between the two methods.

## What was done

1. Anonymous player tokens: position, derived velocity, team sign, offset to the ball, proximity, nearest marker; no player IDs.
2. DeepSets-style shared MLP, mean/max pooling per team and attention pooling overall; combined with 12 global ball/gap/team features.
3. Two-layer bidirectional GRU over 60-second windows, overlapping windows with tent weighting at inference.
4. H/A/D state head and an auxiliary boundary head. Defaults: width 64, hidden 64, 14 training epochs, batch 32, coverage 2, seed 20261108, AdamW and OneCycleLR.
5. Three corruption rounds: ball missing about 41%, player gaps, tracklet permutations per team, duplicate detections. The observation transforms do not read labels.
6. H/A swap + 180-degree rotation augmentation, with H/A labels swapped as well; dead/unknown unchanged.
7. Nested leave-one-match-out on 7 IDSSE matches: 21 pair models and 7 outer models; the decoder is chosen with a validation model that sees neither the outer test match nor the validation match.
8. Objective: proxy spellF1 minus a penalty for count error above 30%; paired bootstrap per match.

## What is and is not supported

Jeremy's report supports the augmentation (+0.055 proxy), the set-temporal model over LightGBM (+0.198), and the decoder over argmax (+0.032). The initial boundary-aware decoding was never selected; gamma 0 was chosen in every fold. A forced boundary cost was about 0.007 lower.

The centered follow-up reduces the problem of a boundary cost that only adds switch penalty, but its delta of 0.001277 has a 95% interval of [−0.000930, 0.003820]; not enough to adopt. The count gate still fails on one match, so adding smoothing is not an automatically safe decision.

## Comparison limits

Jeremy's model is trained and evaluated on degraded IDSSE, whereas Ryu's reference is zero-shot and Farras's earlier models were trained on Metrica event proxies. Proxy numbers from different campaigns are not directly comparable.

Raw target ATD metadata/text, Jeremy's model weights and the IDSSE cache are not committed. The local hardware has CUDA; public IDSSE data can be downloaded from the PySport mirror. The new campaign trains its own adaptation; it does not pretend to use Jeremy's checkpoint or to reproduce 0.581818 exactly.

## Combinations worth testing

Hybrid global features insert Farras's 7 control signals into the set-temporal model. A probability mixture tests whether the control tree helps while the ball is visible and the temporal model helps during gaps. A feature-only GRU separates the benefit of temporal context from the set encoder. Low/asymmetric-cost decoders test under-segmentation; the centered boundary is one limited ablation, not a new large search.

Design, split and selection rules: [PLAN.md](PLAN.md).
