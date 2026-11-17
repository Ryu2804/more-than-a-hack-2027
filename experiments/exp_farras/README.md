# Eksperimen Farras

Eksperimen penguasaan bola di `experiments/`, mengikuti lokasi eksperimen tim. Campaign dan benchmark dipisahkan agar konfigurasi, pilihan kandidat, dan hasil negatif tetap dapat dilacak.

## Metode final tim

**Farras wave5 `short_gru25/dead45` adalah metode final tim untuk kandidat submission dan isi metode laporan teknis.** Penulis laporan mulai dari [spesifikasi lintas branch](../../docs/FINAL_METHOD.md) dan [fakta untuk Typst](../../docs/final_method.json). Identitas/hash runtime: [submission/method.json](../../submission/method.json). Jalur ekspor dan status kesiapan: [submission/README.md](../../submission/README.md).

```powershell
python -B submission/export.py
```

Perintah tersebut memvalidasi dan mengekspor CSV frozen yang dipublikasikan di `submission/artifacts/farras_wave5.csv`, tanpa training, cache lokal atau membaca GT. Bukan jalur inference raw bundle dan tidak mengganti metode dengan public score tertinggi wave6.

## Histori campaign dan riset pembanding

Wave1..6 mempertahankan path dan bukti frozen. Campaign sesudahnya adalah riset dengan protokol tersendiri. Istilah primary/kontrol pada laporan riset tidak menetapkan ulang metode submission.

| Folder | Isi |
|---|---|
| `wave1/` | 8 keluarga metode, 24 konfigurasi, 96 evaluasi eksternal |
| `wave2/` | 16 keluarga metode, 48 konfigurasi, 192 evaluasi eksternal |
| `wave3/` | Kombinasi Jeremy/Farras:11 keluarga probabilitas,66 konfigurasi,594 evaluasi development |
| `wave4/` | Referensi Ryu: PP-TCN lintas provider,64 konfigurasi,1.472 evaluasi development,63 seed fits |
| `wave5/` | Short-possession loss, boundary/masking,96 konfigurasi,2.208 evaluasi development,63 seed fits |
| `wave6/` | Audit6 remote branches,36 alternatif fusion/decoder +1 kontrol,851 evaluasi; tanpa training baru |
| `fair_idsse/` | Benchmark7LOSO folds, training6match sama,63fresh neural seed fits, decoder bersama; bukan ranking exact submission |
| `cross_branch/` | Adaptasi contact Jeremy, memory Rasya, PP-TCN Ryu dan short-loss Farras;63fresh seed fits,28settings |
| `temporal_methods/` | HSMM durasi, consensus change-point dan ordered boundary DP;19settings/133evaluasi, tanpa neural fits baru |
| `spell_training/` | Spell-balanced loss, event sampling, shared-kernel dual-scale TCN;63fresh seed fits,20settings/140evaluasi |
| `event_learning/` | Enam resep lanjutan event sampling;126fresh seed fits,40settings/280evaluasi,42saved-model replays |
| `provider_generalization/` | Source-provider-only transfer dua arah, lima resep, 30 fresh seed fits; sealed label-free inference sebelum 270 metric rows |
| `results/` | Ringkasan skor, konfigurasi, hash kandidat, dan pemeriksaan reproduksi; tanpa label per frame |

Transfer provider baru selesai: kontrol **equal-provider spellF1 0.540864**, RDrop 0.540882 praktis sama; primary modality dropout 0.530378 gagal promosi. Lima resep diuji pada 27 matches, dua conditions. Sebanyak 58 tests lulus/1 skip; 300 artifacts disegel sebelum scoring, 270 probability/decoded arrays dan metric rows direproduksi identik. Tidak ada competition-public/private evaluation baru. Seluruh external matches sudah terekspos historis: **belum ada bukti bebas overfit**, dan angka ini tidak setara dengan skor IDSSE ensemble di bawah. [Hasil dan batas generalisasi](provider_generalization/RESULTS.md), [audit paparan data](provider_generalization/EXPOSURE.md).

Enam resep lanjutan selesai: best exploratory event+contact ensemble **spellF1 IDSSE0.624577**, versus strong research control0.619563. Gain0.005014, menang5/7, tetapi interval masih melintasi0. Primary event+balanced0.619192 gagal; strong control dipertahankan. Alternatif ranking/dead45 mencapai0.621512 dan menang7/7 versus strong consensus control, tetapi perbandingan same-decoder lebih tidak pasti.77tests,42saved-model replays dan280decoded arrays identik. Tidak ada public/private score baru. [Laporan event_learning](event_learning/RESULTS.md).

Training baru selesai: best exploratory event ensemble + consensus **spellF1 IDSSE0.619563**. Primary event ensemble/dead45 mencapai0.618941 versus kontrol0.615458, menang5/7 dengan interval positif, tetapi gain0.003484 belum mencapai batas promosi0.005; kontrol tetap dipakai. Standalone event TCN0.608023 versus short-TCN lama0.595755.27tests,21saved-model inference replays dan140decoded arrays identik. Tidak ada public/private score baru. [Laporan spell_training](spell_training/RESULTS.md).

Campaign metode baru selesai: best exploratory consensus **spellF1 IDSSE0.616970** versus kontrol0.615458, gain0.001512; interval masih melintasi0. Primary HSMM turun0.605838, jadi kontrol tetap dipakai.47semantic tests,133identical decode rebuilds dan7refittedpriors identik. Tidak ada public/private score baru. [Laporan temporal_methods](temporal_methods/RESULTS.md).

Adaptasi lintas branch selesai: primary gabungan **spellF1 IDSSE0.609030**, naik dari kontrol0.595755, menang6/7 dan lolos aturan riset sebelum run. Best raw alternative lebih sederhana, short75+SetSeq25/dead45, mencapai **0.615458**, menang7/7; status eksploratori, bukan pengganti primary post-hoc.31tests lulus,203decoded arrays direproduksi identik. Tidak ada public evaluation baru. [Laporan cross_branch](cross_branch/RESULTS.md), [sumber komponen](cross_branch/SOURCES.md).

Benchmark bersama selesai: **short-loss PP-TCN0.595755**, SetSeq0.578602, PP-TCN0.571090, HGB0.315719 pada mean spellF1 tujuh IDSSE heldout matches. Data/labels/split dan decoder learned arms sama;3seeds/500updates per neural seed. Selisih short-TCN versus SetSeq belum terpisah jelas oleh paired interval. Ini adapted-family benchmark, bukan perbandingan public atau ranking branch final. [Hasil dan batas perbandingan](fair_idsse/RESULTS.md). Laporan juga mengoreksi klaim audit bahwa bug file referensi Ryu otomatis diwarisi feature builder aktual kita.

Wave6 selesai: best hysteresis mencapai **spellF1 development0.674807** dan **diagnostic publik0.786207**, macroF1 publik0.925369. Namun gain development0.004707 belum mencapai batas promosi0.005, sehingga selected tetap wave5 `short_gru25/dead45` (development0.670099, publik0.772414). Best alternative menang7/10 holdout biasa dan6/6 paired ball-drop stress; seluruh data itu sudah terekspos. Detail: [wave6/RESULTS.md](wave6/RESULTS.md); audit semua branch dan bug yang ditemukan: [wave6/BRANCH_REVIEW.md](wave6/BRANCH_REVIEW.md). Private/full-match score belum diketahui. Riwayat: [wave5](wave5/RESULTS.md), [Ryu/wave4](wave4/RESULTS.md), [Jeremy/Farras/wave3](wave3/RESULTS.md).

## Input dan batas publikasi

Kode memanfaatkan cache tracking/nearest-team-distance dari pipeline Ryu. Wave4 menyertakan snapshot5 file referensi Ryu dengan provenance; tidak mengubah eksperimen anggota lain. Cache target belum dapat diregenerasi dari raw bundle resmi dengan kode di folder ini saja.

Secara default, input dibaca dari `experiments/_cache/`:

- `atd_full.npy`, `team_dist.npy`.
- `ext_metrica/Sample_Game_{1,2}_RawTrackingData_{Home,Away}_Team.csv`.
- `ext_metrica/Sample_Game_{1,2}_RawEventsData.csv`.
- `ext_metrica/Sample_Game_3_tracking.txt`, `Sample_Game_3_metadata.xml`, `Sample_Game_3_events.json`.

Data eksternal berasal dari [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data). Jika cache tersedia di lokasi lain, set `FARRAS_SOURCE_EXPERIMENTS` ke **folder induk `_cache`**, bukan ke `_cache` itu sendiri. Folder sumber harus berada dalam workspace induk repo agar manifest dapat menyimpan path relatif tanpa identitas mesin.

Wave3 menambah IDSSE; wave4 menambah20 pertandingan [SkillCorner opendata](https://github.com/SkillCorner/opendata). Wave5 menambah paired ball-drop stress; wave6 memakai probabilitas dan input frozen yang sama tanpa training baru. Prosedur metode utama: [wave5/README.md](wave5/README.md); wave6 tetap riset decoder. Kandidat terpilih wave4/5/6 tidak dilatih pada Metrica; keluarga lama yang memakai anchor tree tetap mewarisi training Metrica.

Contoh PowerShell:

```powershell
$env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\sumber\experiments"
```

Raw tracking, cache fitur, model weights, serta label/frame-level arrays tidak dipush. `out/` dan `cache/` setiap batch diabaikan Git. Folder `results/` hanya memuat metrik agregat, konfigurasi, dan hash. Satu CSV prediksi terpilih dipublikasikan terpisah di `submission/artifacts/farras_wave5.csv` untuk pemeriksaan harness; bukan GT.

## Reproduksi

Untuk metode utama, gunakan [jalur ekspor](../../submission/README.md); untuk reproduksi campaign, ikuti [wave5/README.md](wave5/README.md), termasuk cache/model frozen wave3/4. Ekspor tidak membangun ulang cache dari raw bundle. [wave6/README.md](wave6/README.md) hanya untuk campaign riset decoder pada CPU. Perintah di bawah mempertahankan cara menjalankan wave1/2, bukan jalur submission utama.

Dari root repo, gunakan Python3.10 dan dependencies yang dipin:

```powershell
python -m pip install -r experiments/exp_farras/wave2/requirements.txt
python -B experiments/exp_farras/wave1/data.py
python -B experiments/exp_farras/wave1/run.py all --workers 4
python -B experiments/exp_farras/wave1/run.py public
python -B experiments/exp_farras/wave2/prepare_holdout.py
python -B experiments/exp_farras/wave2/test_decoders.py
python -B experiments/exp_farras/wave2/run.py all --workers 4
python -B experiments/exp_farras/wave2/run.py holdout
python -B experiments/exp_farras/wave2/run.py public
```

Wave2 membutuhkan hasil pemilihan wave1 sebagai pembanding frozen. `all` wave2 menolak menimpa seleksi yang sudah dibekukan. Simpan campaign lama sebelum memulai campaign baru. Jangan memperbarui hash seleksi lama untuk menyamarkan perubahan kode/input.

Menjalankan satu keluarga, misalnya konfigurasi dengan diagnostic public0.56, tanpa mengubah pilihan campaign:

```powershell
python -B experiments/exp_farras/wave2/run.py method --path experiments/exp_farras/wave2/methods/w14_feature_ablation.py
```

Perintah ini menjalankan tiga ablasi dan menghasilkan CSV lokal di `wave2/out/w14_feature_ablation/`. `control.csv` menggunakan7 fitur; tidak memiliki augmentasi pertukaran H/A. Jangan mencampur metode ini dengan kandidat eksternal `w08_symmetric_boost/cost6`.

## Protokol wave1/2

Training dan seleksi memakai Metrica Game1/2, dua fold leave-one-game-out, masing-masing kondisi asli dan tambahan oklusi sintetis. Ranking: mean spellF1 empat kondisi, tie-break macroF1, metode, konfigurasi. Game1/2 digunakan pada kedua batch sehingga validasi development bersifat adaptif.

Sesudah seleksi wave2 dibekukan, hanya kandidat frozen wave1/wave2 dievaluasi pada Game3. Parser Game3 memakai11 format channel metadata dan periode event, bukan slot pemain tetap atau satu babak untuk seluruh pertandingan. GT publik diperiksa setelah freeze, tidak dipakai training atau pemilihan konfigurasi.

Semua model offline5fps, diekspansi25fps. Fitur gap remaining/jendela terpusat menggunakan masa depan. Label eksternal merupakan proxy event, bukan corrected GT dengan definisi identik challenge. Game3 hanya satu pertandingan tambahan.

Detail hasil dan interpretasi: [RESULTS.md](RESULTS.md).
