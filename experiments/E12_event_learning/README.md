# E12: Event learning (six training recipes on top of the event sampler)

Six training recipes on top of the event sampler: balanced team loss, focal team loss, auxiliary boundary supervision, signed change-point ranking, EMA weights, and observed contact inputs. [PLAN.md](PLAN.md) freezes the hypotheses, configurations, the strong research control and the promotion rule before full training. Old folders and other members' experiments are not changed.

Result: the primary `balanced_replace/consensus` reaches 0.619192 against the strong control's 0.619563 (−0.000371) and is **rejected**. The best exploratory `contact_replace/consensus` reaches 0.624577 (+0.005014, wins 5/7), but its interval crosses 0. Record card: [CARD.md](CARD.md). Full results: [RESULTS.md](RESULTS.md).

## Test the result (no match data needed)

Run from the repository root.

1. **Check the recorded numbers** in [CARD.md](CARD.md) against `results/event_learning_results.json`:

   ```bash
   python experiments/check_results.py E12
   ```

   Expected: `PASS` for primary 0.619192, delta −0.000371, not promoted, best contact 0.624577, ranking_replace/dead45 0.621512.

2. **Run the semantic tests** (loss references, boundary/crop semantics, contact parity, EMA, replay rejection):

   ```bash
   python -B -m pytest -q experiments/E12_event_learning
   ```

   Expected: `77 passed` (plus subtests).

## Reproduce the campaign (needs the E08–E11 caches and a GPU)

From the repository root, Python 3.10 with the fair_idsse/spell_training environment, PyTorch, Numba and threadpoolctl. Required: canonical r0 packs/PP features, the audited short/SetSeq OOF probabilities, the cached previous event models/probabilities/replay proofs and the cross_branch raw-25-fps control cache. No data is downloaded and no predecessor weights are loaded for new training.

```bash
python -B experiments/E12_event_learning/check.py
python -B experiments/E12_event_learning/run.py register
python -B experiments/E12_event_learning/run.py fit-fold --match J03WMX --group first
python -B experiments/E12_event_learning/run.py fit-fold --match J03WMX --group second
python -B experiments/E12_event_learning/run.py all
python -B experiments/E12_event_learning/run.py rebuild
```

`check.py` runs the semantic tests, then 18 actual smoke seed fits (1 update per seed, six families). Each smoke ensemble predicts the full held-out match; another process loads the checkpoint and requires identical probabilities. The smoke test is not used as a performance estimate.

`register` freezes the source/PLAN/input/provenance hashes. Full recipe: seven folds, six training matches per fold, three seeds, 500 updates/seed, batch 16, crop 384. Total 126 fresh seed fits / 63,000 updates. `first` runs balanced/focal/boundary; `second` runs ranking/ema/contact; the split is only orchestration, not part of the protocol or the scoring order. All GPU jobs run sequentially in separate child processes. The parent may develop slices in parallel without running several GPU trainings at once.

Every fit stores actual fold/seed/update/parameter audits, model weights and prediction digests. An inference-replay process loads every stored ensemble and checks exact equality with the full held-out probability arrays. `all` verifies/reuses only jobs with an exact identity/hash, completes every fold and then evaluates. Completed results are not overwritten. `evaluate` may be called separately once all 42 jobs and replay proofs are complete.

`rebuild`, in another process, rebuilds the 280 decoded arrays and verifies the 42 model-replay proofs, including byte-identical controls. This is an inference/decode reproduction, not a second full training refit. A non-zero exit is still a failure even if the process wrote some artifacts.

What to compare after a rerun: `rebuild` reports 280 identical arrays and 42 replay proofs; the manifest SHA-256 is `14327721dbf19145c288c54fd95f78564ec427b64d284f070ba1788f8d6095e1`; the 40-configuration table matches [RESULTS.md](RESULTS.md).

## Control and decision

The strong research control uses short 50% + SetSeq 25% + previous event 25% with the frozen consensus decoder (alpha 0.5 / contact_beta 0.5), mean official IDSSE spellF1 0.619562844991005. It is the previous exploratory winner, not a retroactive deployment promotion. The legacy short 75% + SetSeq 25% / dead45 is still reported separately at 0.6154575861548992.

New primary `balanced_replace/consensus`: replace the 25% previous event model with 25% of the event + balanced-loss model. Promotion only if the gain over the strong control is ≥ 0.005, it wins ≥ 5/7 matches and the paired match-bootstrap 95% lower bound is > 0. A raw winner is not chosen automatically when the primary fails. All 40 settings were fixed before the run; fusion/decoder were not retuned from the new results.

Four recipes use the base 135,219 parameters. Boundary adds an auxiliary output; contact adds seven input channels; EMA keeps a training shadow. Equal update counts do not mean full parameter/input/compute parity. The boundary target is computed per full training half before cropping; the original short loss and the complete-spell weighting have different censoring rules, deliberately kept. The boundary auxiliary head is not used by the decoder. EMA is chosen only by the fixed recipe, not by held-out scores.

Every match and hypothesis has been exposed; this is not a fresh blind estimate. The nominal bootstrap intervals do not correct for multiple comparisons or adaptive history. IDSSE results are not compared directly with public/private scores. No target CSV, new public/private evaluation.

## Artifacts and sources

Aggregate exports in `results/event_learning_*.json`. `out/` stores the new checkpoints, raw predictions, audits and replay proofs; all ignored. Raw tracking, labels, weights and per-frame outputs are not shared.

Model, feature and benchmark sources: [spell_training](../E11_spell_training/README.md), [fair_idsse](../E08_fair_idsse/README.md), [cross_branch](../E09_cross_branch/SOURCES.md), [temporal_methods](../E10_temporal_methods/README.md). The actual feature builder uses the correct per-player opponent mask; it does not copy the bug in Ryu's old reference file.
