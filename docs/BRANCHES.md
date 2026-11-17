# Branch map and document status

The final method on every branch is **wave5 `short_gru25/dead45`**, described in [FINAL_METHOD.md](FINAL_METHOD.md) (facts as data in [final_method.json](final_method.json)). Folder names, primary/control terms and "final" labels inside older experiments do not override it.

| Branch | Role after alignment | Material that does not describe the final method |
|---|---|---|
| `main` | Entry point: wave5 implementation and CSV merged from Farras, official rules, final report, and the experiment registry ([experiments/REGISTRY.md](../experiments/REGISTRY.md)) | Competition baselines remain references, not team candidates |
| `farras` | Source of the implementation and of the selected artifact; Farras's research history | Wave1/2, wave6 and later campaigns remain experiments, not replacements for wave5 |
| `ryu` | Archive of Ryu's research and the PP-TCN/feature foundation | `TECHNICAL_DOCUMENT.md`, `writeup.md`, `experiments/RESEARCH.md` and the `015_final_pipeline` folder describe older Ryu variants |
| `jeremy` | Archive of SetSeq, auxiliary-task and recovery research | `exp_1` to `exp_7` use their own campaign protocols/decisions; exp7 was not adopted |
| `rasya` | Archive of rule-based touch/dead-cue research | `TECHNICAL.md` and `CONTEXT.md` are not the final wave5 technical report |
| `consolidation` | Archive of integration/adapters and historical comparisons | `CANDIDATES.md`, `docs/repo_inventory.md` and the common benchmark are not an exact ranking of final candidates |

## Snapshot provenance

The commits below identify the research sources before the documentation alignment. They are **not** the latest HEADs after documentation commits. Branch histories were not rebased, reset, force-pushed or replaced by other members' branches.

| Branch | Research source commit |
|---|---|
| main | `ef22d025b0c02bcdbcec53df34973ece0f53a701` |
| farras | `7254990d82ce5dec3bf1fff6baa4fe2f639e3a60` |
| ryu | `5eaf3e595e91fa0502aac48037d0cae8c356ec6f` |
| jeremy | `ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1` |
| rasya | `a13249724b994728b5686d01bc48e6647cd1e40e` |
| consolidation | `264efa67e1a2e74cce22fea81ed784acf159b258` |

## Reading historical numbers correctly

- Public diagnostics, provider-balanced development, IDSSE-only proxies, frame statistics and private scores are different kinds of result. Do not build one combined leaderboard from different columns.
- Jeremy's exp2/3/5 values of about 0.408/0.410/0.488 in the old registry are predicted target dead shares, not public spellF1. The identified public evidence for Jeremy comes from exp1, not automatically from exp7.
- The old common benchmark used Farras wave2 with limited training, shipped Ryu weights that had already seen the IDSSE benchmark, Rasya's GT-assisted orientation/X mask on external evaluation, and Jeremy's old adapter. That table does not establish an exact quality ranking of submissions or rule compliance.
- Farras's later fair benchmark retrained the model families on a shared split, but it is not an exact reproduction of each branch's final pipeline. The mean advantage of the short loss over SetSeq is not yet clearly separated by its interval.
- Campaign result files are kept as evidence, including negative results. "Archive" means the material must not be used as the final specification, not that every result in it is wrong.

## How the documents relate

Identity, parameters, metric cohorts, attribution and limitations come from [FINAL_METHOD.md](FINAL_METHOD.md). Facts not listed there are unknown, not inferred from a folder name or the best score anywhere. A change to numbers or method points to new artifacts and hashes.

`main` contains the wave5 CSV package and exporter (the submitted file, retrained from scratch, is at the root as `carried_by_admin.csv`). Other research branches refer to the same method without moving or changing their experimental algorithms. The implementation source locked for provenance remains Farras release `7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`.

The Farras integration into `main` was verified with the exporter and the official validator: 199,319 rows, hash identical to the frozen candidate, without `annotation_harness`. [Integration evidence](merge_verification.json). `report_verification.json` records the documentation alignment before this integration, not the hash of the updated branch map.

On `main`, Farras's campaigns were later moved from `experiments/exp_farras/` into numbered folders `experiments/E02_wave1/` … `experiments/E13_provider_generalization/` ([relocation note](../experiments/REGISTRY.md#relocation-note)).
