# Candidate Registry (cross-branch)

## Final team selection

The team selected **Farras wave5 `short_gru25/dead45`**. This decision supersedes the
campaign-local candidate designations below, without replacing their code or evidence.
Use [FINAL_METHOD.md](../docs/FINAL_METHOD.md), [final_method.json](../docs/final_method.json),
and [BRANCHES.md](../docs/BRANCHES.md) as the report specification.

- Pinned source: [`7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60).
- Pinned prediction CSV: [`submission/artifacts/farras_wave5.csv`](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/artifacts/farras_wave5.csv).
- CSV SHA-256: `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.
- Runtime: 75% short-loss PP-TCN ensemble and 25% frozen Farras wave3 SetSequence/GRU, decoded with `dead45`. This branch's wave2 migration adapter is not that implementation.

| Cohort | Metric | Value | Interpretation |
|---|---|---:|---|
| External development | Equal-provider spellF1 | 0.670099305497763 | Selection objective, not a competition score |
| Competition public five-minute window | SpellF1 | 0.7724137931034483 | Diagnostic, not private/leaderboard |
| Competition public five-minute window | MacroF1 | 0.9020586271571484 | Secondary diagnostic metric |
| Competition full-match/private | SpellF1 / macroF1 | Unknown | No private result for this method |

Selection does not certify raw-bundle reproduction, blind evaluation, leakage-free history,
submission readiness, or private superiority. Follow the canonical readiness and exposure notes.

## Historical campaign registry

The following rows preserve earlier research results, not a current leaderboard. Status values
`reference`, `exploratory`, `candidate`, `rejected`, and `ceiling` describe the original campaign.
In particular, `candidate` does not mean selected by the team. External/proxy metrics use
different protocols; deltas and AUC are labeled rather than treated as absolute spellF1.
`-` means not recorded in this registry, not zero. Missing public scores are not inferred.

| Branch | Experiment | Variant | Label set | Historical development/proxy metric | Historical confirm spellF1 | Public spellF1 (diagnostic) | Prediction dead share | Count / mean dur | CSV hash | Historical status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | baseline | `baseline_smooth50` | - | - | - | 0.3256 | - | 26 / 10.8s | - | reference | Public count/duration from repository baseline board; full-match floor 0.2433 is a different cohort |
| jeremy | exp_1 | `setseq_bd` (`setseq_const`) | provider | Proxy spellF1 0.656; reported Δ [0.450,0.487] | - | 0.5818181818181818 | - | 437 / - | `b2d127de…6570` | candidate | Best campaign proxy; count gate fails (-34% worst); public score from `public_sanity.json` |
| jeremy | exp_2 | `exp2_aug` | target_like | Proxy spellF1 0.664 / 0.666 | - | - | 0.4075426326822236 | 378 | `a59128e6…0aed` | candidate | Campaign-local adoption by PREREG H1; former public-column 0.408 was dead share |
| jeremy | exp_3 | `exp3_ref_ens3` | `L_ctrl` | Proxy spellF1 0.748 (`L_ctrl`) | - | - | 0.41040684389104665 | 259 / 14.5s | `17eefbcc…7c08` | candidate | Campaign-local `L_ctrl` option; former public-column 0.410 was dead share |
| jeremy | exp_5 | `exp5_touch_ens3` | provider | Proxy spellF1 Δ +0.010 to +0.015 | - | - | 0.48837004349824054 | 265 / 12.3s | `1b9795cb…42da62` | candidate | Touch gain, provider only; former public-column 0.488 was dead share |
| jeremy | exp_6 | ball recovery | - | Proxy spellF1 Δ -0.125 to -0.145 | - | - | - | - | - | rejected | Detector precision/recall too low |
| farras | wave2 | `w08_symmetric_boost/cost6` | event proxy | Development spellF1 0.5174 | 0.5997 Game3 clean / 0.4340 occ | 0.4606 | - | - | `45e03662…b0446` | candidate | Historical freeze before holdout/public; not wave5 |
| farras | wave2 | `w14_feature_ablation/control` | event proxy | Development spellF1 0.500969 | - | 0.5600 | - | 13 pred poss | - | exploratory | Diagnostic only, not selected |
| rasya | 012 (local report) | touch_hold + gated dead cue | - | Development spellF1 0.5263 / confirm 0.5057 | 0.5057 | 0.4333 (variant mapping unverified) | - | 19 / 9.4s | - | candidate | Secondary registry report, not a pinned native candidate/CSV; campaign-local status |
| rasya | 012 (local report) | + physics filter | - | Development spellF1 delta +0.0063 | - | 0.4824 (secondary report) | - | 17 / 10.4s | - | exploratory | Inventory cites local unpushed commit 37854fb, not current origin/rasya; no reproduced frozen source in this registry |
| rasya | 012 (local) | oracle dead | - | Proxy spellF1 0.6150 | - | - | - | - | - | ceiling | Not deployable |
| ryu | final pipeline (historical name) | TCN + Viterbi | event-derived | Dead AUC 0.90-0.91 (not spellF1) | - | 0.422 | - | 19 / 13.9s | - | candidate | Dead recall 49%, prior shift post-hoc |
| ryu | 017 | multi-match (different training mixes) | event-derived | Proxy spellF1 0.56-0.69 | - | 0.681-0.747 (different variants) | - | - | - | exploratory | Historical public diagnostics; not a public score bound to shipped weights_nometrica |

### Evidence for the Jeremy metric corrections

- [exp_1/results/public_sanity.json](exp_1/results/public_sanity.json): `spell_f1_mean = 0.5818181818181818`, purpose `sanity check, not used for selection`. This is an actual public spellF1 diagnostic.
- [exp_2/results/target_candidate.json](exp_2/results/target_candidate.json): `dead_share = 0.4075426326822236`, also `share_graded.D`; `possession_count = 378`.
- [exp_3/results/target_candidate.json](exp_3/results/target_candidate.json): `dead_share = 0.41040684389104665`, also `share_graded.D`; variant `exp3_ref_ens3`, `possession_count = 259`.
- [exp_5/results/target_candidates.json](exp_5/results/target_candidates.json): `provider.dead_share = 0.48837004349824054`, also `provider.share_graded.D`; variant `exp5_touch_ens3`, `provider.possession_count = 265`.

The last three values are target-prediction label proportions, not comparisons against public
labels. Their evidence files contain no public spellF1; the registry therefore leaves those
public cells unset rather than inventing scores.

## Historical common benchmark (IDSSE, 7 matches)

| Branch | Historical adapter | Mean spellF1 | Mean macroF1 | Mean count error (spells; pred - GT) |
|---|---|---:|---:|---:|
| rasya | touch-hold + gated dead cue | 0.5205 | 0.8539 | -49.0 |
| jeremy | exp_1 SetSeq adapter (legacy) | 0.5980 | 0.8935 | -100.1 |
| farras | wave2 HGB + Viterbi (adapted) | 0.3330 | 0.7955 | -241.0 |
| ryu | TCN + Viterbi (adapted) | 0.5724 | 0.9106 | -179.0 |

These are archived adapter measurements, not a ranking of exact final candidates. Farras used
wave2, not wave5. Ryu's shipped weights had training exposure to IDSSE benchmark matches.
Rasya used GT-assisted external orientation mapping and an `X` mask. Jeremy used the older
exp_1 adapter. See the [benchmark limitations](common_benchmark/README.md).

## Historical campaign rules

These were the consolidation rules, not evidence that every historical run met them:

- Public spellF1 was intended as diagnostic only, not a candidate-selection objective.
- An improvement claim required a CI excluding 0 and majority wins.
- Freeze required a byte-identical rebuild and hash.

For the selected wave5 campaign's actual promotion criteria, reproduction scope, and exposure
limits, use [FINAL_METHOD.md](../docs/FINAL_METHOD.md).