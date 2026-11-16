# Provider generalization

Uji generalisasi source-provider-only, dengan target labels dipisahkan dari proses fit/inference dan seluruh predictions disegel sebelum scoring. Ini **tidak menjamin bebas overfit**: semua27externalmatches pernah terekspos. Audit historis: [EXPOSURE.md](EXPOSURE.md). Resep, split dan keputusan dibekukan di [PLAN.md](PLAN.md).

## Apa yang berbeda

Train7IDSSEr0→test20SkillCornerclean, lalu train20SkillCornerclean→test7IDSSEr0. Fresh weights untuk masing-masing arah; tidak memakai SetSeq/OOFmixtures, shipped weights atau target-rate calibration. Normalization, event indices, short weights dan decoder statistics source-only. Semua match source dan target terpisah per arah, tetapi bukan freshblindmatches secara historis.

Lima resep sama PP-TCN135.219parameter: control event sampler, time-tied PPtoken dropout, one-player-branch modality dropout, training-only MixStyle pooledfeatures, dan two-forward RDrop joint consistency. Masing-masing3seeds/500updates/batch16/crop384. Total30seedfits/15.000updates. RDrop memakai dua forward graphs per update; compute tidak identik.

Setiap target match dinilai pada original condition(`clean`, untuk IDSSEsebenarnya existingr0) dan additional35%ball-drop(`gap`). X16dan PP/ballfeatures diregenerasi dari observasi yang dimask.54conditions berasal dari27matches, bukan54independentmatches. Decoder dead45 tetap dengan rate dari source training saja.

## Menjalankan

Dari root repo, Python3.10 dengan environment fair_idsse/event_learning. Perlu local27externalpacks beserta originalPPfeatures. Tidak mengunduh rawdata, memakai providedtargetmatch atau membuka competitionGT.

```powershell
python -B experiments/exp_farras/provider_generalization/prepare.py
python -B experiments/exp_farras/provider_generalization/check.py
python -B experiments/exp_farras/provider_generalization/run.py register
python -B experiments/exp_farras/provider_generalization/run.py all-job --source idsse --family control
python -B experiments/exp_farras/provider_generalization/run.py all
python -B experiments/exp_farras/provider_generalization/run.py verify
```

`prepare.py` memisahkan observation carriers dan external-label vault tanpa menghitung metric. Newcache memakai folder sendiri; sourcepacks/oldcampaigns tidak diubah. CompetitionpublicGTread ditolak runtime guard, termasuk deliberateprobe yang tidak membaca bytes.

`check.py` menjalankan semantic tests, lalu15real smoke seedfits masing-masing1update pada7sourceIDSSEmatches. Inference dan isolatedsaved-modelreplay memakai satu SkillCornermatch pada duaconditions melalui label-freecarriers. Target-label vault dan original label-bearing testpacks ditolak. Smoke tidak menilai metric.

`register` membekukan source/PLAN/input digests setelah checks. `all-job` menjalankan fit,predict,replay pada tiga proses berbeda. Fit hanya dapat membuka source-labeledpacks. Predict/replay memakai observation-onlycarriers dan tidak dapat membuka original labeledpacks/vault melalui guardedPythonopen. GPUjobs berurutan. `all` melanjutkan seluruh10jobs, melakukan seal lengkap lalu score. Jangan menjalankan score sebagian atau membuat ulang recipe setelah hasil terlihat.

Untuk orchestration bertahap setelah10jobs selesai:

```powershell
python -B experiments/exp_farras/provider_generalization/run.py seal
python -B experiments/exp_farras/provider_generalization/run.py score
python -B experiments/exp_farras/provider_generalization/run.py verify
```

Seal write-once mencakup270predictionNPZs,10checkpoints,10metadata dan10replayproofs. Seluruh300artifacts wajib lengkap dan hashcocok. Scorer memverifikasi seal sebelum membuka externallabelvault. `verify` pada proses lain mengulang270metricrows dari sealedpredictions, bukan fulltrainingrefit. Fullmodelreplay sudah menuntut270jointprobability/decodedarrays identik. Completedresults tidak ditimpa.

## Decision dan batas bukti

Primary `modality_dropout` versus `control`, original condition, objective equal mean dua target providers. Promosi membutuhkan gain>=0.01, tidak ada provider turun lebih dari0.02, dan provider-stratified paired-bootstrap95%lower>0. Bootstrap5.000draws, seed20261120. Jika gagal, control dipertahankan; gap diagnostic dan raw winner tidak memilih primary pengganti. Semua recipe fixed, tanpa public-target tuning.

Runtimeguard adalah trustedPythonread policy, **bukan OSsandbox**. Tidak menutup seluruhnativeI/O/existingdescriptors/adversarialbypass. Report expectedprobe terpisah dari unexpectedblockedreads; yangterakhir fatal. Hashseal menjamin integritas artifact yang diverifikasi, bukan membuktikan tidakoverfit. Transfer provider juga mengubah label/proxydefinitions,trackingquality dan coverage. Nominalbootstrap tidak mengoreksi adaptivehistory/multiplecomparisons.

Hasil lengkap dan keputusan: [RESULTS.md](RESULTS.md). Agregat di `../results/provider_generalization_*.json`. cache/outignored menyimpan labelvault,observations,features,weights,predictions,stressspans dan seal; raw/deriveddata tersebut tidak dipublish. Tidak ada targetCSV, public/privateevaluation baru, commit atau pushautomatis.
