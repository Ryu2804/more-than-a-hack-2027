# Pinned branch components and adaptation limits

A fresh fetch before the campaign found no new commits on the six remote branches. Components were chosen by their adopted results and a matching label definition, not by taking the largest number from different protocols.

## Jeremy

Pinned `ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1`.

- [exp5 report](https://github.com/Ryu2804/more-than-a-hack-2027/blob/ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1/experiments/exp_5/REPORT.md): adopted provider touch-only ensemble0.674 target_like, delta+0.0102 versus reference. exp4's higher0.679 was selection winner, weaker on fresh-seed replication; not renamed as adopted exp5 result.
- [Contact implementation](https://github.com/Ryu2804/more-than-a-hack-2027/blob/ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1/experiments/exp_3/exp3_touch.py): two team minimum distances, peak velocity change, team at peak over preceding5rawframes. Reimplemented as observation-only features with half/missingness safeguards; no original global-constant monkeypatch.
- Existing fair SetSeq probabilities retained as complementary architecture, but these are fresh500step benchmark OOF fits, not exp5 weights. New contact channels are learned by TCNs, not retrained GRUs.
- exp7 recovery robustness did not pass all adoption clauses; detector failed independent precision/recall gates. Do not introduce noisy recovered-ball arrays or modify GT into L_ctrl just to inherit higher headline scores.

## Ryu

Pinned `5eaf3e595e91fa0502aac48037d0cae8c356ec6f`.

- [017 multi-match notes](https://github.com/Ryu2804/more-than-a-hack-2027/blob/5eaf3e595e91fa0502aac48037d0cae8c356ec6f/experiments/017_multi_match/NOTES.md): PP encoder improved grouped30match mean0.661 to0.679. Full30match public0.747 belongs a different model/protocol.
- Adapt player-set attention/pooled-player TCN and conditional heads using existing actual wave4 implementation. Add seven new observation-control channels through original control_tcn API; no copying whole source pipeline.
- Shipped27match weights are not used: they saw all IDSSE matches. New contextual models exclude test match and share500step/3seed budget.
- Important audit correction: defect in reference-copy features is not automatically a defect of actual local wave4/features.py; actual builder already uses per-player opponent teams. Neither old source nor old feature caches are patched here.

## Rasya

Pinned `a13249724b994728b5686d01bc48e6647cd1e40e`.

- [CONTEXT](https://github.com/Ryu2804/more-than-a-hack-2027/blob/a13249724b994728b5686d01bc48e6647cd1e40e/CONTEXT.md): adopted C06 touch_hold radius0.8m/min30frames, out>=15frames or nearest speed<2m/s>=25frames, suppress dead runs<150frames. Reported0.5263dev/0.5057confirm.
- Adapt full25fps touch memory, decaying recency and dead cue as learned input features. Native rule diagnostic included separately, not forced into neural labels.
- Never use heldout GT to flip teams, impose dead labels, or copy X mask. Speeds derive from observed positions/elapsedseconds. Before first touch/after half boundary memory is unknown, not backfilled from future team.
- Reported longslow challenger0.5504/0.5413 was below adoption threshold, not automatically selected. No oracle scores treated as deployable performance.

## Farras

Remote `87f3236d4ed875f4cc868d9f750a0776a7779333` still has old waves1/2. Local fair benchmark is newer and is the control chain for this campaign.

- Original fair short-loss PP-TCN mean0.5957554001098855 on seven heldout IDSSE matches. Reuse only exact audited OOF probabilities from matching heldout jobs.
- Combine with SetSeq and new contact+memory TCN using fixed joint weights. Primary .50short+.25SetSeq+.25combined; no weights learned on outer folds.
- Fixed wave6-style dead hysteresis low0.35/high0.50 compared against dead45. No public0.786207 tuning or re-evaluation. Original fair baseline and short GT retained.

## Main and consolidation

Main `ef22d025b0c02bcdbcec53df34973ece0f53a701`: official mean spellF1 scorer, H/A/D/X conventions and published baseline. Main is not a competing learned research model.

Consolidation `264efa67e1a2e74cce22fea81ed784acf159b258`: common benchmark objective, not old adapter rankings or mislabeled dead shares. Six-match exclusion and same labels/metric are enforced here; source docs remain unchanged.

## Interpretation

Contact/memory models add sub-step25fps observation information beyond the5fps baseline tensors. This is an intentional information/feature experiment on the same source stream, not input-tensor parity or exact best-submission reproduction. The three new ablation models alone have identical architecture, dimension and budget. Seven matches and decoder hypotheses are historically exposed, so confidence intervals remain exploratory. No claim about private score is supported.
