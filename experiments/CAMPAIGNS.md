# Farras's campaigns (E02–E13)

Ball-possession experiments in `experiments/`, following the team's experiment layout. Campaigns and benchmarks are kept separate so configurations, candidate choices and negative results stay traceable. Each campaign has its own folder with a `CARD.md` and a `README.md` explaining how to test its results; the order is in [REGISTRY.md](REGISTRY.md).

## Team final method

**Farras wave5 `short_gru25/dead45` is the team's final method for the submission candidate and the method section of the technical report.** Report authors start from the [cross-branch specification](../docs/FINAL_METHOD.md) and the [Typst facts](../docs/final_method.json). Runtime identity/hash: [submission/method.json](../submission/method.json). Export path and readiness status: [submission/README.md](../submission/README.md).

```bash
python -B submission/export.py
```

This command validates and exports the frozen CSV published in `submission/artifacts/farras_wave5.csv`, without training, local caches or reading GT. It is not a raw-bundle inference path, and it does not replace the method with wave6's higher public score.

## Campaign history and comparison research

Wave1..6 (E02–E07) keep their frozen evidence; the folders were moved from `experiments/exp_farras/` (see the [relocation note](REGISTRY.md#relocation-note)). Later campaigns are research with their own protocols. Primary/control terms in research reports do not redefine the submission method.

| Folder | Contents |
|---|---|
| [`E02_wave1/`](E02_wave1/) | 8 method families, 24 configurations, 96 external evaluations |
| [`E03_wave2/`](E03_wave2/) | 16 method families, 48 configurations, 192 external evaluations |
| [`E04_wave3/`](E04_wave3/) | Jeremy/Farras combination: 11 probability families, 66 configurations, 594 development evaluations |
| [`E05_wave4/`](E05_wave4/) | Ryu reference: cross-provider PP-TCN, 64 configurations, 1,472 development evaluations, 63 seed fits |
| [`E06_wave5/`](E06_wave5/) | Short-possession loss, boundary/masking, 96 configurations, 2,208 development evaluations, 63 seed fits |
| [`E07_wave6/`](E07_wave6/) | Audit of 6 remote branches, 36 fusion/decoder alternatives + 1 control, 851 evaluations; no new training |
| [`E08_fair_idsse/`](E08_fair_idsse/) | Benchmark with 7 LOSO folds, same 6 training matches, 63 fresh neural seed fits, shared decoder; not an exact submission ranking |
| [`E09_cross_branch/`](E09_cross_branch/) | Adapting Jeremy's contact, Rasya's memory, Ryu's PP-TCN and Farras's short loss; 63 fresh seed fits, 28 settings |
| [`E10_temporal_methods/`](E10_temporal_methods/) | Duration HSMM, consensus change-point and ordered boundary DP; 19 settings / 133 evaluations, no new neural fits |
| [`E11_spell_training/`](E11_spell_training/) | Spell-balanced loss, event sampling, shared-kernel dual-scale TCN; 63 fresh seed fits, 20 settings / 140 evaluations |
| [`E12_event_learning/`](E12_event_learning/) | Six further event-sampling recipes; 126 fresh seed fits, 40 settings / 280 evaluations, 42 saved-model replays |
| [`E13_provider_generalization/`](E13_provider_generalization/) | Two-way source-provider-only transfer, five recipes, 30 fresh seed fits; sealed label-free inference before 270 metric rows |
| `EXX_*/results/` | Score summaries, configurations, candidate hashes and reproduction checks per campaign; no per-frame labels |

Provider transfer finished: control **equal-provider spellF1 0.540864**, RDrop 0.540882 practically the same; the primary, modality dropout at 0.530378, failed promotion. Five recipes were tested on 27 matches in two conditions. 58 tests passed / 1 skipped; 300 artifacts were sealed before scoring, and 270 probability/decoded arrays and metric rows were reproduced identically. No new competition public/private evaluation. All external matches had been exposed historically: **there is no evidence of being overfit-free**, and these numbers are not comparable with the IDSSE ensemble scores below. [Results and generalisation limits](E13_provider_generalization/RESULTS.md), [data exposure audit](E13_provider_generalization/EXPOSURE.md).

Six further recipes finished: best exploratory event+contact ensemble **IDSSE spellF1 0.624577**, versus strong research control 0.619563. Gain 0.005014, wins 5/7, but the interval still crosses 0. The primary, event+balanced at 0.619192, failed; the strong control was kept. The ranking/dead45 alternative reached 0.621512 and won 7/7 against the strong consensus control, but the same-decoder comparison is less certain. 77 tests, 42 saved-model replays and 280 decoded arrays identical. No new public/private score. [event_learning report](E12_event_learning/RESULTS.md).

New training finished: best exploratory event ensemble + consensus **IDSSE spellF1 0.619563**. The primary, event ensemble/dead45, reached 0.618941 versus control 0.615458, winning 5/7 with a positive interval, but the gain of 0.003484 did not reach the 0.005 promotion threshold; the control was kept. Standalone event TCN 0.608023 versus the old short-TCN 0.595755. 27 tests, 21 saved-model inference replays and 140 decoded arrays identical. No new public/private score. [spell_training report](E11_spell_training/RESULTS.md).

New-method campaign finished: best exploratory consensus **IDSSE spellF1 0.616970** versus control 0.615458, gain 0.001512; the interval still crosses 0. The primary HSMM dropped to 0.605838, so the control was kept. 47 semantic tests, 133 identical decode rebuilds and 7 identical refitted priors. No new public/private score. [temporal_methods report](E10_temporal_methods/RESULTS.md).

Cross-branch adaptation finished: combined primary **IDSSE spellF1 0.609030**, up from control 0.595755, winning 6/7 and passing the research rule set before the run. The simpler best raw alternative, short75+SetSeq25/dead45, reached **0.615458**, winning 7/7; it is exploratory, not a post-hoc replacement for the primary. 31 tests passed, 203 decoded arrays reproduced identically. No new public evaluation. [cross_branch report](E09_cross_branch/RESULTS.md), [component sources](E09_cross_branch/SOURCES.md).

Shared benchmark finished: **short-loss PP-TCN 0.595755**, SetSeq 0.578602, PP-TCN 0.571090, HGB 0.315719 on mean spellF1 over seven held-out IDSSE matches. Data/labels/split and decoder were the same for the learned arms; 3 seeds / 500 updates per neural seed. The difference between short-TCN and SetSeq is not clearly separated by the paired interval. This is an adapted-family benchmark, not a public comparison or a final branch ranking. [Results and comparison limits](E08_fair_idsse/RESULTS.md). The report also corrects the audit claim that a bug in Ryu's reference file was automatically inherited by our actual feature builder.

Wave6 finished: best hysteresis reached **development spellF1 0.674807** and **public diagnostic 0.786207**, public macroF1 0.925369. But the development gain of 0.004707 did not reach the 0.005 promotion threshold, so the selection stays wave5 `short_gru25/dead45` (development 0.670099, public 0.772414). The best alternative won 7/10 ordinary holdout scenarios and 6/6 paired ball-drop stress cases; all of that data had been exposed. Details: [E07_wave6/RESULTS.md](E07_wave6/RESULTS.md); audit of all branches and the bugs found: [E07_wave6/BRANCH_REVIEW.md](E07_wave6/BRANCH_REVIEW.md). The private/full-match score is unknown. History: [wave5](E06_wave5/RESULTS.md), [Ryu/wave4](E05_wave4/RESULTS.md), [Jeremy/Farras/wave3](E04_wave3/RESULTS.md).

## Inputs and publication limits

The code uses the tracking/nearest-team-distance cache from Ryu's pipeline. Wave4 includes a snapshot of 5 Ryu reference files with provenance; it does not modify other members' experiments. The target cache cannot yet be regenerated from the official raw bundle with the code in these folders alone.

By default, inputs are read from `experiments/_cache/`:

- `atd_full.npy`, `team_dist.npy`.
- `ext_metrica/Sample_Game_{1,2}_RawTrackingData_{Home,Away}_Team.csv`.
- `ext_metrica/Sample_Game_{1,2}_RawEventsData.csv`.
- `ext_metrica/Sample_Game_3_tracking.txt`, `Sample_Game_3_metadata.xml`, `Sample_Game_3_events.json`.

External data comes from [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data). If the cache lives elsewhere, set `FARRAS_SOURCE_EXPERIMENTS` to **the parent folder of `_cache`**, not to `_cache` itself. The source folder must be inside the workspace that contains the repository, so manifests can store relative paths without machine identity.

Wave3 added IDSSE; wave4 added 20 [SkillCorner open data](https://github.com/SkillCorner/opendata) matches. Wave5 added the paired ball-drop stress test; wave6 used the same frozen probabilities and inputs without new training. Main-method procedure: [E06_wave5/README.md](E06_wave5/README.md); wave6 remains decoder research. The selected wave4/5/6 candidates were not trained on Metrica; older families that use the anchor tree still inherit Metrica training.

Example:

```bash
export FARRAS_SOURCE_EXPERIMENTS=/path/to/workspace/experiments      # PowerShell: $env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\experiments"
```

Raw tracking, feature caches, model weights and label/frame-level arrays are gitignored. Each campaign's `out/` and `cache/` are ignored by Git. The `results/` folders contain only aggregate metrics, configurations and hashes. One selected prediction CSV is published separately in `submission/artifacts/farras_wave5.csv` (and as `submission.csv` at the root) for harness checks; it is not GT.

## Reproduction

For the main method, use the [export path](../submission/README.md); to reproduce a campaign, follow [E06_wave5/README.md](E06_wave5/README.md), including the frozen wave3/4 caches and models. The export does not rebuild caches from the raw bundle. [E07_wave6/README.md](E07_wave6/README.md) covers only the CPU decoder-research campaign. The commands below keep the original way of running wave1/2; they are not the main submission path.

From the repository root, with Python 3.10 and the pinned dependencies:

```bash
python -m pip install -r experiments/E03_wave2/requirements.txt
python -B experiments/E02_wave1/data.py
python -B experiments/E02_wave1/run.py all --workers 4
python -B experiments/E02_wave1/run.py public
python -B experiments/E03_wave2/prepare_holdout.py
python -B experiments/E03_wave2/test_decoders.py
python -B experiments/E03_wave2/run.py all --workers 4
python -B experiments/E03_wave2/run.py holdout
python -B experiments/E03_wave2/run.py public
```

Wave2 needs wave1's selection as a frozen comparison. Wave2's `all` refuses to overwrite a frozen selection. Save an old campaign before starting a new one. Never update an old selection hash to hide a code or input change.

Running one family, for example the configuration with the 0.56 public diagnostic, without changing the campaign choice:

```bash
python -B experiments/E03_wave2/run.py method --path experiments/E03_wave2/methods/w14_feature_ablation.py
```

This runs three ablations and writes local CSVs to `E03_wave2/out/w14_feature_ablation/`. `control.csv` uses 7 features and has no H/A swap augmentation. Do not mix this method with the external candidate `w08_symmetric_boost/cost6`.

## Wave1/2 protocol

Training and selection use Metrica Games 1/2, two leave-one-game-out folds, each in its original condition and with added synthetic occlusion. Ranking: mean spellF1 over the four conditions, tie-break macroF1, then method, then configuration. Games 1/2 are used in both batches, so development validation is adaptive.

After the wave2 selection was frozen, only the frozen wave1/wave2 candidates were evaluated on Game 3. The Game 3 parser handles 11 metadata channel layouts and event periods, rather than fixed player slots or a single half for the whole match. The public GT was checked after the freeze and was not used for training or configuration selection.

All models are offline at 5 fps and expanded to 25 fps. The gap-remaining and centred-window features use the future. External labels are event proxies, not corrected GT with the challenge's exact definition. Game 3 is only one additional match.

Detailed results and interpretation: [E03_wave2/RESULTS.md](E03_wave2/RESULTS.md).
