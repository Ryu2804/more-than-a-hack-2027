# Cross-branch: contact, memory dan ensemble temporal

[PLAN.md](PLAN.md) menetapkan primary `triple_combined/hysteresis` sebelum full fits, dibanding original fair control `short/dead45`. Tidak mengubah hasil/caches eksperimen teman atau campaign lama.

Selesai: primary **spellF1 IDSSE0.609030**, kontrol0.595755, gain0.013274, menang6/7; lolos aturan promosi riset. Best raw alternative short75+SetSeq25/dead45 lebih tinggi0.615458, menang7/7, tetapi tetap exploratory.31semantic tests,63fresh seed fits dan203identical decode rebuilds. Tidak ada public evaluation baru. [Laporan lengkap](RESULTS.md).

## Adaptasi

- Ryu: PP-TCN, view symmetry, player-set attention, conditional dead/team probabilities.
- Jeremy: contact globals dari observasi25fps (exp3/4/5 touch-only adopted recipe) serta cached SetSeq OOF dari benchmark bersama.
- Rasya: label-free last-touch hold, contact recency, sustained out/slow cue, suppression dead runs<6s. Tanpa GT flipping, oracle dead/X mask, atau salah satuan speed.
- Farras: cached short-loss PP-TCN, fixed hysteresis decoder dan joint-probability ensembles.
- Main/consolidation: official scorer dan common evaluation; tidak menyalin ranking adapter lama yang tidak sebanding.

Ini transfer komponen penelitian, bukan reproduksi exact best submission tiap branch. Cached SetSeq bukan model touch-exp5 asli. Fitur contact/memory dipindahkan ke TCN GPU agar tidak mengulang training SetSeq CPU yang lama; efeknya diukur lewat tiga ablation model berarsitektur/input-channel-count/budget sama.

## Protokol

Tujuh IDSSE r0 yang sama dengan fair_idsse,7LOSO folds. Tiap model baru training6match lainnya,3seeds0/1/2,500updates/seed,batch16,crop384. Contact-only,memory-only,combined memakai36inputchannels dan PP encoder yang sama; channel ablation diisi0.63fresh seed fits/31500updates. Cached baseline probabilities diambil hanya pada heldout match sesuai metadata six-match exclusion, manifest dan input hashes.

Raw25fps observasional di-corrupt ulang dengan seed canonicalr0, labels dibuang sebelum regeneration. Sampled positions/ball/metadata/timeline wajib cocok dengan prepared5fps. New controls menambah informasi di antara sampled frames: ini feature augmentation, bukan tensor-input parity dengan baseline lama.

Empat belas probability families x2fixeddecoders=28settings/196per-match evaluations. Native raw25fps rule sampled5fps dilaporkan terpisah. Primary saja boleh lolos promotion untuk riset (gain>=.005,>=5/7wins,paired-bootstrap95%lower>0); best raw alternative tidak otomatis mengganti primary. Tidak ada public evaluation atau submission target baru. Data dan hypotheses sebelumnya sudah terekspos, bukan fresh blind test.

## Menjalankan

Dari root repo, Python3.10 dengan dependency/environment preceding fair_idsse. Perlu clean25fps records lokal, canonicalr0 packs, PP feature packs, dan hasil/caches OOF fair_idsse yang lengkap serta immutable. Tidak mengunduh data atau membuka old weights.

```powershell
python -B -m unittest discover -s experiments/exp_farras/cross_branch -p 'test_*.py' -v
python -B experiments/exp_farras/cross_branch/prepare.py
python -B experiments/exp_farras/cross_branch/check.py
python -B experiments/exp_farras/cross_branch/run.py register
python -B experiments/exp_farras/cross_branch/run.py fit-fold --match J03WMX
python -B experiments/exp_farras/cross_branch/run.py all
python -B experiments/exp_farras/cross_branch/run.py rebuild
```

`check.py` benar-benar melatih ketiga model baru1step/seed pada6realtrainingmatches dan memprediksi match lain. Ini smoke, tidak dicampur ke performance estimate500steps. Parent memperlakukan proses exit nonzero sebagai failure walaupun output sudah tertulis. Fit memakai child process terpisah setiap family/fold. `all` memverifikasi/reuse hanya job yang exact identity-nya cocok dan menolak menimpa results yang sudah lengkap. Sumber/PLAN/preparedcontrols dibekukan sebelum full fits; jangan memperbarui hash untuk menyamarkan perubahan.

`rebuild` menjalankan ulang203decoded arrays dalam proses lain (196model-setting rows+7native-rule rows), memakai frozen probabilities/context. Ini decode rebuild, bukan model refit.

## Artefak dan batas publikasi

`cache/` menyimpan controls/native predictions per frame, `out/` menyimpan audits, preregistration, models/probabilities, hasil lengkap dan verification. Semua ignored. Hanya kode, dokumen dan aggregate JSON di `../results/cross_branch_*.json` yang dibagikan. Raw tracking, labels/probabilities, weights dan submissions tidak dipush. Tidak ada commit/push otomatis.

IDSSE CC BY4.0; provenance dan kredit sebelumnya: [fair benchmark](../fair_idsse/README.md), [Jeremy reference](../wave3/reference/SOURCE.md), [Ryu reference](../wave4/reference/SOURCE.md). Source refs lengkap dibekukan pada registry; catatan source decisions ada di PLAN.

Rincian branch/commit, hasil adopted versus exploratory, serta bagian yang sengaja tidak diimpor: [SOURCES.md](SOURCES.md).
