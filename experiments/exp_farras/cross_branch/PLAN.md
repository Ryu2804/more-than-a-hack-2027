# Cross-branch adaptation: contact, possession memory, PP-TCN and short-loss ensemble

## Source decisions before fitting

Fresh fetch: all six remote refs unchanged. main ef22d025, farras87f3236, jeremyccd7b8c5, rasyaa1324972, ryu5eaf3e59, consolidation264efa67. Main contributes official scoring; consolidation contributes common-evaluation goal, not its noncomparable ranking. Actual learned/rule ideas come from four research branches:

- Ryu: player-set attention + pooled-player TCN and two conditional heads; use audited actual wave4 builder (opponent relative each player), not defective reference-copy feature.
- Jeremy: strongest adopted provider recipe is exp5_touch_ens3, touch-only replication (+0.0102 provider proxy), rather than non-adopted exp7 recovery simulation. Adopt true25fps contact globals from exp3_touch/exp4/5 and cached independent SetSeq OOF probabilities. No new slow CPU GRU fit; cached SetSeq comes from previous fair benchmark and is NOT exact exp5 weights.
- Rasya: adopt last-touch hold and sustained out/slow cue from validated C06, with 1.2s minimum carried run and6s dead-sliver suppression. Remove GT team flipping and GT D/X masks. Speeds derive from observed positions in actual source meters/25fps; never divide by3.6. Do not force weak rule outputs onto neural labels.
- Farras: retain complete-short-spell weighted TCN OOF probabilities (fair benchmark best mean0.595755), plus wave6 fixed hysteresis low.35/high.50. Do not treat public0.786207 as new training label or selection objective.

This is a new adaptation experiment, not byte-exact reproductions of teammates' best submissions. Source scores use different data/label/decoder protocols and are not a rankable joint leaderboard. Shared IDSSE budget/split remains authoritative.

## Exact data and fresh model protocol

Same seven IDSSE r0 packs/labels as completed fair benchmark, same7LOSO folds. Each new model trains on exactly6other matches; no SkillCorner/Metrica/target/public access. Provider GT unmerged, including all short possessions. Sampling and normalization only training fold. Three new GPU model families: contact_tcn, memory_tcn, combined_tcn. All original control_tcn architecture36channels plus player-set encoder, same500updates/seeds0,1,2/batch16/crop384/optimizer/masking/noise as preceding PP-TCN. Total63fresh seed fits/31500updates. None loads old weights. Original short/SetSeq/PP OOF probabilities may be reused only with exact heldout identity, six disjoint train names, manifest/input/prediction hashes. No final weights inferred on training matches.

Compute control features from full25fps locally cached clean observation records, remove labels before corruption, regenerate with canonical r0seed20261108+originalmatchindex. Regenerated sampled normalized positions/ball/team vector must match prepared5fps packs before features accepted. No downloads or writes to old caches. This restores sub-step contact information not available from differentiating sampled5fps positions.

## Seven new feature channels, same dimension in all ablations

Jeremy contact0..3: preceding5rawframes in samehalf, per-team minimum player-ball distance /10 cap3; peak norm(second ball difference)*25 /20 cap3, valid three consecutive observed ball positions; team nearest at peak (+1home/-1away/0unknown). Missing distance cap3; undefined peak0. Equal maxima earliest index. No provider state/speed/possession labels.

Rasya context4..6: heldteam sign (home+1/away-1/none0), exp(-seconds since last observed touch/10) or0 beforefirsttouch, corrected out/slow deadcue after removing cueD runs<150rawframes. Touch radius.8m; carried team runs<30rawframes absorbed into previous team without future fill at half start. Out cue requires15rawframes, nearest-player speed<2m/s requires25; finite visible ball and metadata-valid player required. All durations/derivatives reset at half boundaries. Native rule hard codes kept as diagnostic, not force-applied to neural prediction.

Views swap distance columns0/1 and negate peakteam3/heldteam4; other channels unchanged. contact_tcn zeros4..6, memory_tcn zeros0..3, combined_tcn usesall7. Existing player/ball/PP arrays not changed. Original training ball mask also zeros all new control channels, preventing visible contacts from surviving intentional synthetic masking. Source raw metadata determines teams, not held-out agreement.

## Fixed combinations and decoding

Fourteen families: short,setseq,pp,contact,memory,combined; five .75short+.25other mixtures; triple_contact and triple_combined (.50short+.25SetSeq+.25newmodel); quality_combined with newmodelweight .10+.30*mean11(ballvisibility), remainingmassshort. Joint probabilities mixed before conditional heads; centered window clipped within half. No learned calibrator, label-based gate, provider special case or tuned outer-fold weight.

Each family has two fixed decoders: original fair dead45, or hysteresis low.35/high.50/min5sample component. Both H/A live reset, same18lambda grid, training-fold rate, merge1s and calibration after merge. All28settings reported. Native25fps Rasya diagnostic sampled to5fps separately; not ranked as same probabilistic decoder.

Primary hypothesis declared BEFORE fitting: triple_combined/hysteresis versus short/dead45. For a research promotion it must gain>=0.005, win>=5/7matches and paired-bootstrap95% lower>0 (5000draws seed20261115, unique-match units). No fallback selecting best alternative when primary fails. Overall best raw alternative is exploratory only. Control must reproduce mean0.5957554001098855 exactly. No new public evaluation or submission generation.

## Exposure and integrity

All seven matches and decoder ideas have prior research exposure. OOF prevents training on the heldout match, but this is adaptive hypothesis development, not fresh blind validation. No fake nested-CV claim: no learned meta model or outer-label weight selection. Bootstrap is exploratory/unadjusted, not family-wise inference.

Parent creates source/input/config registry after actual tests and prepare, before full new fits. It includes previous fair source/input hashes, previous selected results/manifest, reused OOF probabilities+metadata, local clean25fps records, regenerated controls, source/degrader digests, new PLAN/code. Every child must exit0; no accepting files after failed native teardown. GPU TCNs only; CPU SetSeq probabilities reused, not refit.

Run in distinct child per family/fold. First fold then remaining folds so progress can be reported without waiting through entire training silently. Independent process repeats all28x7 decoder outputs and native7 arrays from frozen probabilities, compares label hashes. This is decoding rebuild, not model refit.

Code/report/results only in new cross_branch plus parent indexes and safe aggregate JSON exports. Old campaigns/teammate files immutable. No commit/push automatically; raw tracking, labels/probabilities, weights and submissions ignored. IDSSE CC BY4.0; source credits follow preceding fair benchmark and pinned GitHub refs.

Input-resource distinction: prepared5fps base tensors tetap sama, tetapi contact/memory baru menambah ringkasan raw25fps observasional di antara sampled frames. Ini feature-information augmentation yang sengaja diuji, bukan klaim seluruh model menerima tensor identik. Semua berasal dari sumber/corruption stream yang sama dan tidak menambah dataset/labels. Cached baselines tidak dilatih ulang agar menerima informasi baru; tiga contextual TCN ablations adalah pembanding yang benar-benar berdimensi dan berbudget identik.
