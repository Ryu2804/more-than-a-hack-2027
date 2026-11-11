# Candidate Registry (cross-branch)

Status values: `reference`, `exploratory`, `candidate`, `frozen`, `rejected`, `ceiling`.

| Branch | Experiment | Variant | Label set | Proxy spellF1 | Confirm spellF1 | Public spellF1 (diagnostic) | Count / mean dur | CSV hash | Status | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| main | baseline | `baseline_smooth50` | - | - | - | 0.3256 | 16 / 18.5s | - | reference | Published floor (full-match 0.2433) |
| jeremy | exp_1 | `setseq_bd` (`setseq_const`) | provider | 0.656 [0.450,0.487] Δ | - | 0.582 | 437 / - | `b2d127de…6570` | candidate | Best proxy; count gate fails (-34% worst) |
| jeremy | exp_2 | `exp2_aug` | target_like | 0.664 / 0.666 | - | 0.408 | 378 | - | candidate | Adopted by PREREG H1 |
| jeremy | exp_3 | `exp3_ref_ens3` | `L_ctrl` | 0.748 (`L_ctrl`) | - | 0.410 | 259 / 14.5s | `17eefbcc…7c08` | candidate | Adopted `L_ctrl` option |
| jeremy | exp_5 | `exp5_touch_ens3` | provider | +0.010 to +0.015 | - | 0.488 | 265 / 12.3s | `1b9795cb…42da62` | candidate | Touch gain, provider only |
| jeremy | exp_6 | ball recovery | - | -0.125 to -0.145 | - | - | - | - | rejected | Detector precision/recall too low |
| farras | wave2 | `w08_symmetric_boost/cost6` | event proxy | 0.5174 dev | 0.5997 Game3 clean / 0.4340 occ | 0.4606 | - | `45e03662…b0446` | candidate | Frozen before holdout/public |
| farras | wave2 | `w14_feature_ablation/control` | event proxy | 0.500969 dev | - | 0.5600 | 13 pred poss | - | exploratory | Diagnostic only, not selected |
| rasya | 012 (local) | touch_hold + gated dead cue | - | 0.5263 dev / 0.5057 confirm | 0.5057 | 0.4333 base | 19 / 9.4s | - | candidate | High dead recall ~0.97 |
| rasya | 012 (local) | + physics filter | - | +0.0063 dev | - | 0.4824 | 17 / 10.4s | - | exploratory | Public diagnostic, no pre-reg |
| rasya | 012 (local) | oracle dead | - | 0.6150 | - | - | - | - | ceiling | Not deployable |
| ryu | final pipeline | TCN + Viterbi | event-derived | dead AUC 0.90-0.91 | - | 0.422 | 19 / 13.9s | - | candidate | Dead recall 49%, prior shift post-hoc |
| ryu | 017 | multi-match | event-derived | 0.56-0.69 | - | 0.681-0.747 | - | - | exploratory | Public diagnostic only |

## Rules
- Public spellF1 is diagnostic only and never selects a candidate.
- A claim of improvement requires CI excluding 0 and majority wins.
- Freeze requires byte-identical rebuild and hash.