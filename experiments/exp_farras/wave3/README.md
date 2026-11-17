# Wave3: kombinasi temporal-set dan kontrol bola

Desain dibekukan di [PLAN.md](PLAN.md). Referensi Jeremy dipin pada commit `0e35aae54e09ca744441681fb076f656bbc88b5a`; sumber parser/degrader dipertahankan verbatim dan diatribusikan di [reference/SOURCE.md](reference/SOURCE.md).

Campaign ini **adaptasi terkontrol**, bukan reproduksi byte-identik model Jeremy. Model lebih kecil, training8 epoch, token dari observasi5fps, lima pertandingan development dan dua IDSSE holdout. Skor Jeremy0.581818 dan diagnostic Farras0.56 hanya angka publik historis, bukan target tuning.

Campaign selesai. Kandidat terpilih `hybrid_tree25/centered2`: public spellF1 **0.575610**, refit199.319 baris identik. Laporan, ranking66 konfigurasi, dan batas hasil: [RESULTS.md](RESULTS.md).

## Persiapan

Jalankan dari root repo. Python3.10 dan dependensi `requirements.txt`. Runtime terpasang PyTorch2.13.0+cu126, tetapi campaign memakai CPU single-thread: backend cuDNN GRU lokal crash saat proses ditutup. CPU dan native CUDA lolos lifecycle smoke; CPU lebih cepat daripada native CUDA pada smoke yang sama. Model/epoch tidak dikurangi. Tidak memakai layanan training eksternal.

```powershell
$env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\sumber\experiments"
$env:FARRAS_TRAIN_DEVICE = "cpu"
python -B experiments/exp_farras/wave3/prepare_idsse.py
python -B experiments/exp_farras/wave3/prepare_local.py
python -B experiments/exp_farras/wave3/test_invariants.py
python -B experiments/exp_farras/wave3/run.py dev
python -B experiments/exp_farras/wave3/run.py holdout
python -B experiments/exp_farras/wave3/run.py public
```

`prepare_idsse` mengunduh metadata/tracking tujuh pertandingan publik (~2.62GB total) dari [mirror PySport](https://huggingface.co/datasets/pysport/idsse-data); ukuran dan LFS SHA-256 diverifikasi. IDSSE CC BY4.0, Bassek et al.2025, [paper](https://doi.org/10.1038/s41597-025-04505-y), [data](https://doi.org/10.6084/m9.figshare.28196177). Unduhan dan data turunan tetap lokal, tidak dipush.

`prepare_local` membutuhkan cache Ryu dan Metrica yang sama dengan wave1/2. Jalankan persiapan wave1 dan wave2 terlebih dahulu bila cache sebelumnya belum tersedia; petunjuk di [README induk](../README.md). Override menunjuk folder induk `_cache`. Tidak membutuhkan raw target GT. Team mapping target dipulihkan dari nearest-distance cache hanya bila bukti memenuhi ambang dan minima hasil rekonstruksi cocok; tidak mengklaim membaca metadata XML yang belum tersedia.

## Pemilihan dan artefak

- `cache/data/`: observasi, fitur dan proxy labels; tidak dipush.
- `cache/jeremy/`: snapshot audit branch, tidak dipush.
- `cache/predictions/`: probabilitas fold, tidak dipush.
- `out/preregistered.json`: konfigurasi dan hash kode/input sebelum training.
- `out/models/`: fitted checkpoints lokal.
- `out/selection.json`, `external_scoreboard.csv`: ranking66 konfigurasi hanya berdasarkan development external.
- `out/holdout_report.json`: empat kandidat prespecified, empat skenario holdout, dan reproduksi independen kandidat target.
- `out/candidates/*.csv`: empat CSV target lokal, tidak dipush.
- `out/public_audit.json`: diagnostic setelah freeze untuk empat kandidat saja.

Training diserialkan pada CPU single-thread. Worker hanya mengimplementasikan komponen. Harness menjalankan fit dan pemeriksaan prediksi. `dev` dapat melanjutkan cache yang selesai dengan hash preregistrasi yang sama. Setelah seleksi dibekukan, jangan menjalankan ulang campaign dengan kode yang diubah atau menimpa bukti lama.

## Batas

IDSSE proxy memiliki dead-ball/coverage berbeda dari challenge. Metrica menggunakan event-derived labels. Dua holdout IDSSE belum dipakai training campaign ini, tetapi pernah dipakai Jeremy dan hasil agregatnya memengaruhi desain. Metrica3 pernah dievaluasi pada wave2. Diagnostic publik tidak membuktikan skor private/full-match.

Prediksi offline memakai konteks dua arah dan gap remaining. Dataset5fps menurunkan resolusi batas possession. Cost/arsitektur/configuration grid tidak diperluas setelah skor diketahui. Hasil negatif dan possession-count error tetap dilaporkan.
