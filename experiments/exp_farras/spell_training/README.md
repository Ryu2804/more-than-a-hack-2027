# Spell training

Eksperimen training baru untuk possession pendek: spell-balanced team loss, short-event-centered sampling, dan shared-kernel dual-scale TCN. Folder ini terpisah dari eksperimen anggota lain dan seluruh campaign terdahulu. [PLAN.md](PLAN.md) menetapkan konfigurasi, primary dan aturan promosi sebelum full fits.

## Menjalankan

Dari root repo, Python3.10 dengan environment fair_idsse, PyTorch/Numba/threadpoolctl, canonical r0packs dan PPfeature packs yang sama. Perlu cached OOF short_tcn/SetSeq beserta provenance fair_idsse, controls cross_branch dan hasil/rebuild temporal_methods. Tidak mengunduh data dan tidak membuka predecessor model weights.

```powershell
python -B experiments/exp_farras/spell_training/check.py
python -B experiments/exp_farras/spell_training/run.py register
python -B experiments/exp_farras/spell_training/run.py fit-fold --match J03WMX
python -B experiments/exp_farras/spell_training/run.py all
python -B experiments/exp_farras/spell_training/run.py rebuild
```

`check.py` menjalankan semantic tests, lalu9actualseedfits masing-masing1update pada enam realtrainingmatches dan fullheldout inference untuk tiga keluarga. Tiga proses terpisah memuat ulang smoke checkpoints dan memeriksa probabilities identik. Ini smoke, bukan performance estimate.

`register` membekukan source/PLAN/input/provenance hashes setelah checks. `fit-fold` menjalankan setiap model family pada child process terpisah, tiga seeds/500updates per family, kemudian proses lain memuat weights baru dan mengulang fullheldout inference. Proses exit nonzero tetap failure meski sudah menulis artefak. GPU jobs berurutan; paralelisasi pengembangan tidak berarti trainingGPU bersamaan. `all` melanjutkan hanya job yang exact identity/hash/budget-nya cocok dan tidak menimpa completed results.

`evaluate` dapat dipanggil setelah tujuh folds selesai; menilai20setting combinations pada setiap match. `rebuild` membangun ulang140decoded arrays pada proses lain dan memverifikasi21saved-model inference proofs. Ini bukan full training refit kedua.

## Perbandingan

Kontrol: short-loss TCN75% + SetSeq25%, dead45, mean official spellF1 IDSSE0.6154575861548992. Primary baru: event_replace25/dead45, yaitu short50% + SetSeq25% + event-sampledTCN25%. Masing-masing keluarga juga dinilai pure, base75new25 danreplace25, dengan frozen dead45 serta frozen consensus decoder. Seluruh kombinasi ditentukan sebelum run, bukan scan setelah melihat hasil.

Same7LOSO folds/6trainingmatches,3seeds,500updates,batch16,crop384,29channels dan parametercount. Dual-scale memakai kernel yang sama untuk dilation1dan dilation asli; parameter tidak bertambah, convolution compute bertambah. Spell loss dan event sampling memang mengubah objective/window distribution, bukan mengklaim tensor/cost parity penuh.

Keputusan promosi primary: gain>=0.005, menang>=5/7matches, paired match-bootstrap95%lower>0. Jika gagal, kontrol tetap dipakai; best raw exploratory tetap dilaporkan. Match dan hipotesis sudah pernah terekspos, bukan fresh blind test. Tidak ada public/private evaluation baru atau target submission. Tidak ada commit/push otomatis.

## Artefak

Hasil lengkap: [RESULTS.md](RESULTS.md). Aggregate JSON di `../results/spell_training_*.json`. `out/jobs/` menyimpan weights baru/probabilities/metadata, `out/replays/` menyimpan exact inference proofs. Raw training/labels/weights/probabilities tidak dipublikasikan; seluruh `out/` ignored.

Provenance komponen: [fair_idsse](../fair_idsse/README.md), [cross_branch](../cross_branch/SOURCES.md), [temporal_methods](../temporal_methods/README.md). Model dasar PP-TCN mengikuti referensi Ryu yang sudah dipisahkan dari sumber aslinya; feature builder aktual mempertahankan per-player opponent mask yang benar. Semua original source tetap immutable.
