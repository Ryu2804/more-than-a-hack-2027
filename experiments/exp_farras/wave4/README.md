# Wave4: referensi Ryu, TCN lintas provider

[PLAN.md](PLAN.md) mengunci split,64 konfigurasi, batas perbandingan, dan aturan pemilihan sebelum training. Source Ryu dipin pada commit `5eaf3e595e91fa0502aac48037d0cae8c356ec6f`; [reference/SOURCE.md](reference/SOURCE.md) memuat kredit dan provenance.

Laporan Ryu terbaru menyebut public spellF1 sekitar0.747 untuk player-set TCN, sedangkan technical document lama masih0.422. Jangan mencampur versi tersebut atau menganggap checkpoint no-Metrica/cached-input adapter pasti menghasilkan angka sama.

Hasil terukur: kandidat eksternal `pp_gru25/no_shift` mendapat **spellF1 publik0.772414**, sama dengan PP-TCN murni. Adapter shipped Ryu0.760000; wave3 frozen0.575610. Holdout SkillCorner6 membaik, tetapi Metrica3 lebih rendah daripada wave3. Seluruh hasil, batas perbandingan, dan hash refit: [RESULTS.md](RESULTS.md).

## Menjalankan

Dari root repo, Python3.10 dengan dependency wave3, CUDA Conv1d bila smoke runtime lolos. Metrica/target/IDSSE caches wave3 diperlukan; raw SkillCorner diunduh dari upstream publik berlisensi MIT. Tidak ada upload data atau training remote.

```powershell
$env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\sumber\experiments"
$env:FARRAS_TRAIN_DEVICE = "cuda"
python -m pip install -r experiments/exp_farras/wave4/requirements.txt
python -B experiments/exp_farras/wave4/prepare_skillcorner.py
python -B experiments/exp_farras/wave4/prepare_features.py
python -B experiments/exp_farras/wave4/prepare_prior.py dev
python -B experiments/exp_farras/wave4/test_invariants.py
python -B experiments/exp_farras/wave4/run.py dev
python -B experiments/exp_farras/wave4/prepare_prior.py holdout
python -B experiments/exp_farras/wave4/run.py holdout
python -B experiments/exp_farras/wave4/run.py public
```

`prepare_prior` berjalan pada CPU di proses terpisah, karena checkpoint wave3 memakai modul `models` sendiri dan backend GRU CUDA lokal gagal saat shutdown. Campaign TCN memakai CUDA hanya setelah actual Conv1d+PP fitting/refit/checkpoint smoke selesai bersih. `FARRAS_TRAIN_DEVICE=cpu` dapat dipilih eksplisit sebelum preregistrasi; setelah freeze backend harus tetap sesuai registry.

Jangan menjalankan ulang `prepare_prior dev` setelah preregistrasi karena file NPZ input ikut dihash. `dev` dapat melanjutkan prediksi/checkpoint yang sudah selesai dengan source/input yang identik, tetapi menolak menimpa seleksi frozen. Untuk perubahan algoritme, buat campaign baru, bukan mengganti hash hasil lama.

## Data dan validasi

Training baru hanya19 match:5 IDSSE +14 SkillCorner. IDSSE memakai3 corruption rounds wave3, SkillCorner memakai native broadcast tracking dan stochastic masking saat training. Enam grouped folds mengecualikan seluruh round match yang sama. Metrica1/2 dipakai zero-shot stress validation, tidak melatih TCN baru.

Holdout campaign:2 IDSSE +6 SkillCorner +Metrica3 asli/degraded. IDSSE/Metrica3 sudah pernah dilihat dalam campaign sebelumnya; SkillCorner6 baru untuk Farras tetapi tidak baru untuk riset Ryu. Ryu shipped weights telah dilatih pada semua27 IDSSE/SkillCorner sehingga tidak dinilai sebagai held-out pada pertandingan itu. Adapter shipped reference hanya Metrica3 dan target.

Bobot3 seed untuk tiap TCN ensemble dan prior GRU/tree menjadi input bagi8 probability families. Delapan decoder fixed menghasilkan64 konfigurasi. Metric utama mean spellF1 dengan bobot provider seimbang; macroF1 tie-breaker. Hanya empat kandidat fixed dievaluasi publik setelah freeze, tidak ada64-way public search.

## Artefak

- `cache/raw/`: raw SkillCorner lokal, tidak dipush.
- `cache/data/`, `cache/features/`, `cache/prior/`, `cache/predictions/`: prepared data dan probabilitas, tidak dipush.
- `cache/ryu_weights/`: checkpoint source Ryu, tidak dipush ulang.
- `out/preregistered.json`, `out/selection.json`, `out/external_scoreboard.csv`: registry/freeze64 konfigurasi.
- `out/models/`:3 fitted TCN ensembles lokal.
- `out/holdout_report.json`: evaluasi fixed references, adaptasi source Ryu, dan independent-refit hash.
- `out/public_audit.json`: empat diagnostic publik, bukan dasar penggantian kandidat.
- `out/candidates/*.csv`: label per frame, tetap lokal.

Kode wave1/2/3 dan branch anggota lain tidak diubah. Tidak ada commit/push otomatis. Source asli, data provider dan cache raw tetap memiliki provenance sendiri. PP-TCN dan GRU kandidat terpilih tidak dilatih pada Metrica; keluarga yang memakai anchor tree lama tetap mewarisi training Metrica beserta ketidakjelasan lisensinya. Metrica juga dipakai untuk evaluasi zero-shot.
