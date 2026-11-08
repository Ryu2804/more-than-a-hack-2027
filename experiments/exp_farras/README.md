# Eksperimen Farras

Eksperimen penguasaan bola di `experiments/`, mengikuti lokasi eksperimen tim. Dua batch dipisahkan agar konfigurasi, pilihan kandidat, dan hasil negatif tetap dapat dilacak.

| Folder | Isi |
|---|---|
| `wave1/` | 8 keluarga metode, 24 konfigurasi, 96 evaluasi eksternal |
| `wave2/` | 16 keluarga metode, 48 konfigurasi, 192 evaluasi eksternal |
| `results/` | Ringkasan skor, konfigurasi, hash kandidat, dan pemeriksaan reproduksi; tanpa label per frame |

Hasil diagnostic publik tertinggi: **spellF1 0.560000**, menggunakan histogram gradient boosting dengan 7 fitur kontrol bola dan Viterbi berpenalti pergantian 6. Kandidat ini **bukan pilihan validasi eksternal**. Kandidat wave2 yang dibekukan sebelum holdout/public menggunakan augmentasi pertukaran H/A dan mencapai public spellF1 0.460606. Tidak ada klaim skor private/full-match.

## Input dan batas publikasi

Kode memanfaatkan cache tracking/nearest-team-distance dari pipeline Ryu. Tidak menyalin pipeline Ryu atau mengubah eksperimen anggota lain. Cache target belum dapat diregenerasi dari raw bundle resmi dengan kode di folder ini saja.

Secara default, input dibaca dari `experiments/_cache/`:

- `atd_full.npy`, `team_dist.npy`.
- `ext_metrica/Sample_Game_{1,2}_RawTrackingData_{Home,Away}_Team.csv`.
- `ext_metrica/Sample_Game_{1,2}_RawEventsData.csv`.
- `ext_metrica/Sample_Game_3_tracking.txt`, `Sample_Game_3_metadata.xml`, `Sample_Game_3_events.json`.

Data eksternal berasal dari [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data). Jika cache tersedia di lokasi lain, set `FARRAS_SOURCE_EXPERIMENTS` ke **folder induk `_cache`**, bukan ke `_cache` itu sendiri. Folder sumber harus berada dalam workspace induk repo agar manifest dapat menyimpan path relatif tanpa identitas mesin.

Contoh PowerShell:

```powershell
$env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\sumber\experiments"
```

Raw tracking, cache fitur, model weights, CSV submission, serta label/frame-level arrays tidak dipush. `out/` dan `cache/` setiap batch diabaikan Git. Folder `results/` hanya memuat metrik agregat, konfigurasi, dan hash.

## Reproduksi

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

## Protokol

Training dan seleksi memakai Metrica Game1/2, dua fold leave-one-game-out, masing-masing kondisi asli dan tambahan oklusi sintetis. Ranking: mean spellF1 empat kondisi, tie-break macroF1, metode, konfigurasi. Game1/2 digunakan pada kedua batch sehingga validasi development bersifat adaptif.

Sesudah seleksi wave2 dibekukan, hanya kandidat frozen wave1/wave2 dievaluasi pada Game3. Parser Game3 memakai11 format channel metadata dan periode event, bukan slot pemain tetap atau satu babak untuk seluruh pertandingan. GT publik diperiksa setelah freeze, tidak dipakai training atau pemilihan konfigurasi.

Semua model offline5fps, diekspansi25fps. Fitur gap remaining/jendela terpusat menggunakan masa depan. Label eksternal merupakan proxy event, bukan corrected GT dengan definisi identik challenge. Game3 hanya satu pertandingan tambahan.

Detail hasil dan interpretasi: [RESULTS.md](RESULTS.md).
