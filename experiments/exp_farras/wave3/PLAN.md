# Wave3: identity-free context + control evidence

## Evidence and goal

Pinned Jeremy branch: `0e35aae54e09ca744441681fb076f656bbc88b5a`. His report gives public spellF1 0.581818, seven-match IDSSE proxy0.656, macroF1 around0.918. His best arm uses identity-free player sets + a two-layer BiGRU, corruption resampling, and team/field mirroring. Constant-cost Viterbi was selected instead of boundary-aware decoding. A centered-boundary follow-up improved mean proxy by0.0013 with interval crossing0 and was not adopted. Possession-count quality gate failed on one match.

Our earlier public diagnostic maximum0.56 uses seven control features, histogram boosting, Viterbi6, trained on Metrica1/2. Our externally selected wave2 candidate scores0.460606 public and was verified on Metrica3. These proxy scores do not share a dataset with Jeremy and are not directly comparable.

Goal: test whether Jeremy's identity-free temporal context complements Farras control evidence, under matched observations/folds. Do not optimize the known public window. No claim of reproducing Jeremy's0.656: his nested protocol, training14 epochs, 25fps-derived tokens, and final decoder differ from this adapted controlled campaign.

## Inputs and split, before fitting

Acquire the public IDSSE XML metadata/tracking from the PySport mirror, verifying inventory sizes and LFS SHA-256. Use the unchanged Jeremy parser and exact label-free25fps degrader, three rounds per match, seeds20261108+1000*round+match_index. IDSSE license CC BY4.0, Bassek et al.2025, https://doi.org/10.1038/s41597-025-04505-y . Mirror https://huggingface.co/datasets/pysport/idsse-data . Raw data and frame labels remain local.

Development IDSSE: J03WMX,J03WN1,J03WOH,J03WOY,J03WPY. Locked campaign holdout: J03WQQ,J03WR9. Jeremy previously used all seven, and his aggregate findings influenced the design; these are unseen for this campaign's training/selection, not a globally blind study.

Metrica1/2 are cross-provider development stress matches, never neural/IDSSE-tree training data. Metrica3 clean/degraded are a secondary holdout stress test, already exposed in wave2 and explicitly not fresh. Proxy event labels differ from provider possession labels and from challenge corrected labels.

Target uses existing ATD and nearest-distance caches, not GT. Raw metadata absent; active5fps slot teams may be recovered only when nearest-distance votes are sufficiently pure and reconstructed team minima match the source cache. Unresolved present slots must fail, not be guessed. Empty sampled slots remain masked. Disclose recovered cache provenance instead of claiming XML metadata validation.

## Models

All models share prepared normalized positions, observation-only16 features, and5fps blocks. No test labels enter feature/prediction APIs. Unknown labels ignored; blocks reset at each half.

| Family | Fixed change |
|---|---|
| setseq | Jeremy-style masked home/away mean/max + attention pooling; two-layer BiGRU |
| hybrid | Same model plus seven scaled Farras control features in globals |
| feature_gru | Two-layer BiGRU on the16 hand features; no set encoder |
| control | Histogram gradient boosting on seven control features, IDSSE training |
| control_symmetric | Same tree with H/A swap augmentation |
| all_features | Histogram gradient boosting on all16 features, IDSSE training |
| anchor | Previous Metrica-trained Farras control model; same observation packs at prediction |
| hybrid_tree25/50/75 | Arithmetic mixture of hybrid and anchor, tree weight0.25/0.5/0.75 |
| hybrid_visibility | Tree weight0.75 when ball visible,0.25 otherwise |

Neural training fixed seed2027, window300 steps (60s), batch16, epochs8, coverage1, width32, hidden32, lr0.002, AdamW weight_decay0.0001, boundary loss weight0.5 and positive weight3. Set models use mirroring and all three corruption rounds; feature_gru no mirroring. No early stopping. Fits serialized, CPU single-thread. This is deliberately a new controlled training recipe, not byte reproduction of Jeremy's final checkpoint.

Runtime decision before campaign training: cuDNN GRU on local Windows/PyTorch2.13.0+cu126 repeatedly produced deterministic predictions but exited with code -1073740791 during process teardown, including without checkpoint roundtrip. Native CUDA GRU and CPU GRU both completed cleanly; CPU completed the same smoke faster than native CUDA. Campaign therefore uses explicit `FARRAS_TRAIN_DEVICE=cpu`, without suppressing crashes or changing model architecture/epochs. Runtime choice is recorded in preregistration.

Tree recipe fixed max_leaf_nodes15, max_iter100, lr0.08, l2=1, no early stopping, max40k rows and balanced weights. Learned boundary head uses only current/previous observations and training state changes. Anchor has no learned boundary head; neutral boundary probability0.5.

## Decoder experiments: six fixed choices

| Name | H/A switch cost | D transition cost | Centered boundary strength |
|---|---:|---:|---:|
| cost05 | 0.5 | 0.5 | 0 |
| cost2 | 2 | 2 | 0 |
| cost6 | 6 | 6 | 0 |
| team05_dead2 | 0.5 | 2 | 0 |
| team2_dead05 | 2 | 0.5 | 0 |
| centered2 | 2 | 2 | 0.5 |

Temperature1. Centered transition cost `max(0,lambda - gamma*(logit(boundary_t)-logit(mean_boundary_in_block)))`. Unlike uncentered boundary cost, high boundary confidence can lower switch penalty. State stays cost0. This remains an unproven hypothesis, not an assumed improvement.

11 probability families x6 decoders =66 configurations. No grid expansion after observing scores.

## Selection and references

Five IDSSE leave-one-match-out folds. Train other four matches with r0/r1/r2, evaluate held match r0. Final model fit on all five predicts Metrica1/2 clean and r0 zero-shot. Average two conditions inside each Metrica match, then equally average five IDSSE and two Metrica match spellF1 values. Tie-break mean macroF1, family name, decoder name. Count error is reported as a quality warning at30%, not substituted for the competition metric.

Anchor trains only Metrica1/2 using the original cached clean/occlusion copies; it excludes the evaluated Metrica match. It uses no IDSSE labels. Old frozen symmetric histogram model is a fixed reference, not an extra selected family.

After freeze evaluate only selected candidate plus three prespecified references: adapted setseq/cost05; anchor/cost6; previous symmetric histogram/cost6. Holdout consists of the last two IDSSE matches and Metrica3 clean/r0. No training/parameter/selection updates based on holdout. Evaluate public GT only afterward, for those four fixed candidates, not a66-way public search.

Verify selected target prediction by independent refit with fixed inputs/seed. Save code/input hashes, fitted checkpoint hashes, predictions privately, and aggregate results for sharing. Historical public0.56 and Jeremy0.581818 stay diagnostic reference numbers, not selection objectives.

## Execution ownership

Parent owns plan, harness, model/decoder registry, execution, freeze, verification, and interpretation. Seven workers own only acquisition adapter, local-data adapter, tokens, temporal models, trees, decoder, and fusion respectively. Workers do not run training, choose parameters, read target GT, or alter prior campaigns.
