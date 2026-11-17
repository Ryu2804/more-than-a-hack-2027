# Experiment registry

Every experiment on `main`, in the order it was run. Each experiment is one folder `EXX_<name>/`
holding its code, its `results/*.json`, its own `PLAN.md` / `RESULTS.md` where they exist, and a
`CARD.md` record card (question, design, decision rule, result, verdict). Cards are
retrospective: they summarise the campaign's own files and add nothing that is not in them.

**Final team method: [E06](E06_wave5/CARD.md), Farras wave5 `short_gru25/dead45`.**
Authoritative spec: [docs/FINAL_METHOD.md](../docs/FINAL_METHOD.md). Nothing below replaces it.

Order is established by lineage: each campaign's control is the previous campaign's
selected or best result (column *Builds on*). Experiments on the `ryu`, `jeremy`, `rasya` and
`consolidation` branches are not registered here; see [docs/BRANCHES.md](../docs/BRANCHES.md).

| ID | Experiment | Builds on | Question | Primary result | Verdict |
|---|---|---|---|---|---|
| [E00](E00_eda/CARD.md) | [`E00_eda/`](E00_eda/) | n/a | What does the target tracking look like, and what does it imply for design? | Ball missing on 40.7% of graded frames; median gap 10.7 s | Shaped all later designs |
| [E01](E01_baseline/CARD.md) | [`E01_baseline/`](E01_baseline/) | E00 | How does the official nearest-player + vote baseline behave? | smooth50: public spellF1 0.325581, full match 0.2433 | Reference floor |
| [E02](E02_wave1/CARD.md) | [`E02_wave1/`](E02_wave1/) | E01 | Which rule/learned family works on external (Metrica) data? | `m01_hysteresis/r15`: dev 0.460517 | Selected (superseded) |
| [E03](E03_wave2/CARD.md) | [`E03_wave2/`](E03_wave2/) | E02 | Do learned tree models + Viterbi beat rules? | `w08_symmetric_boost/cost6`: dev 0.517438 | Selected (superseded) |
| [E04](E04_wave3/CARD.md) | [`E04_wave3/`](E04_wave3/) | E03 | Does Jeremy's identity-free set+BiGRU complement Farras control features? | `hybrid_tree25/centered2`: dev 0.615394 | Selected (superseded) |
| [E05](E05_wave4/CARD.md) | [`E05_wave4/`](E05_wave4/) | E04 | Do Ryu's PP-TCN and cross-provider training help? | `pp_gru25/no_shift`: dev 0.656490 | Selected (superseded) |
| [E06](E06_wave5/CARD.md) | [`E06_wave5/`](E06_wave5/) | E05 | Can short-possession loss, boundary head or masking beat wave4? | `short_gru25/dead45`: dev 0.670099 (+0.013609) | **Promoted: final method** |
| [E07](E07_wave6/CARD.md) | [`E07_wave6/`](E07_wave6/) | E06 | Can fusion/decoder changes alone beat wave5? | `m05_dead_hysteresis/low35_high50`: dev 0.674807 (+0.004707) | Not promoted (< 0.005) |
| [E08](E08_fair_idsse/CARD.md) | [`E08_fair_idsse/`](E08_fair_idsse/) | E06 | Which model family is best under one shared split and decoder? | short-loss PP-TCN 0.595755 (IDSSE LOSO) | Benchmark only |
| [E09](E09_cross_branch/CARD.md) | [`E09_cross_branch/`](E09_cross_branch/) | E08 | Do all branches' cues combined beat short-loss PP-TCN? | primary 0.609030; best raw short75+SetSeq25 0.615458 | Research primary passed |
| [E10](E10_temporal_methods/CARD.md) | [`E10_temporal_methods/`](E10_temporal_methods/) | E09 | Do duration-aware / consensus decoders beat dead45? | HSMM primary 0.605838 (−0.009620) | Rejected |
| [E11](E11_spell_training/CARD.md) | [`E11_spell_training/`](E11_spell_training/) | E10 | Does spell-targeted training (loss, sampling, dual-scale) help? | `event_replace25/dead45` 0.618941 (+0.003484) | Not promoted (< 0.005) |
| [E12](E12_event_learning/CARD.md) | [`E12_event_learning/`](E12_event_learning/) | E11 | Do six event-sampling recipes beat the strong control? | primary 0.619192 (−0.000371) | Rejected |
| [E13](E13_provider_generalization/CARD.md) | [`E13_provider_generalization/`](E13_provider_generalization/) | E12 | Do regularisers improve transfer to an unseen provider? | modality dropout 0.530378 (−0.010486) | Rejected |

## Relocation note

Until commit `efdb82e` the Farras campaigns lived in `experiments/exp_farras/<campaign>/` with one
shared `exp_farras/results/`. They were moved here with `git mv` (history is preserved: use
`git log --follow`). Path constants in the code were updated so every unit-test suite still passes,
and `submission/export.py` still reproduces the frozen CSV hash. Paths and source hashes **recorded
inside** `results/*.json` and the pre-registered `PLAN.md` files still refer to the old layout, so
each campaign's `check.py` hash comparison will flag the moved/edited files. The frozen pre-move
state is the release commit `7254990` (final method) and `efdb82e` (all campaigns).
`reference/` snapshots and the English `PLAN.md` files of E04, E05 and E10–E13 were left byte-identical. The Indonesian `PLAN.md` files of E06–E09 were replaced by English versions, so their recorded pre-registration hashes no longer match; the originals are in git history at `efdb82e` under `experiments/exp_farras/<campaign>/PLAN.md`.

## Testing a result

Every `EXX_*/README.md` starts with a **Test the result** section. The first step needs no match data:

```bash
python experiments/check_results.py          # recompute every card's headline numbers from results/*.json
python -B -m pytest -q experiments/E06_wave5  # unit tests of one experiment (where it has tests)
```

Later steps (rerunning a notebook or a campaign) need the data bundle and the gitignored caches.

## Reading the numbers

- **Columns are not one leaderboard.** "dev" is each campaign's own external-development
  objective on its own cohort (Metrica only in E02–E03, IDSSE + Metrica in E04, three providers in
  E05–E07, 7 IDSSE leave-one-match-out folds in E08–E12, cross-provider transfer in E13). Compare
  numbers only inside one card.
- **Unit of replication is the match.** Frames are not independent samples, and corruption rounds
  or stress versions of a match are repeated measures of the same unit. Paired bootstrap
  intervals in the cards resample unique matches.
- **Public 5-minute GT (17 possessions) is a post-freeze diagnostic**, never a selection signal.
  The full-match (private) score of every experiment is unknown.
- **Exposure.** All external holdout matches had been seen by earlier campaigns. Results are
  adaptive research, not fresh blind tests ([exposure audit](E13_provider_generalization/EXPOSURE.md)).

## Adding an experiment

1. Take the next ID, create `EXX_<name>/`, and copy [_template/CARD.md](_template/CARD.md) to `EXX_<name>/CARD.md`. Add a `README.md` with the steps to test the result.
2. Fill *Question*, *Hypothesis*, *Design* and *Decision rule* **before** fitting, and commit them.
3. Run inside `EXX_<name>/` (outputs to its ignored `out/` and `cache/`; aggregate JSON to `results/`);
   after the run, fill *Result*, *Verdict* and *Threats to validity*, and add its headline numbers to `check_results.py`.
4. Add one row to the table above.
