# Metode final tim: Farras wave5

Tim menetapkan `short_gru25/dead45` sebagai metode final. Penulis laporan memakai [spesifikasi lintas branch](../docs/FINAL_METHOD.md) dan [fakta untuk Typst](../docs/final_method.json). Identitas runtime kandidat, hash CSV, dan bukti seleksi tetap dikunci di [method.json](method.json). Campaign baru tidak otomatis mengganti pilihan ini.

## Ekspor CSV terpilih

Dari root repo, Python 3.10 dan dependencies evaluasi:

```powershell
python -m pip install -r requirements.txt
python -B submission/export.py
```

Output: `submission/out/farras.csv`. Untuk nama tim yang berbeda:

```powershell
python -B submission/export.py --output submission/out/teamname.csv
```

Perintah ini memeriksa hash bukti seleksi dan CSV, menjalankan validator resmi, lalu menyalin byte kandidat frozen. Tidak melakukan training, scoring, membaca GT, atau memilih ulang kandidat. File tujuan yang berbeda tidak ditimpa; ekspor berulang ke file identik tetap aman.

CSV prediksi terpilih disertakan di Git untuk pemeriksaan harness:

```text
submission/artifacts/farras_wave5.csv
```

Fresh clone dapat membaca CSV tersebut atau menjalankan ekspor tanpa cache, model, raw data, maupun GT. File ini adalah output algoritme frozen wave5, bukan ground truth. Tidak ada CSV kandidat lain yang dipublikasikan. Prosedur campaign dan dependensi cache/model untuk training/inference ulang tetap di [wave5/README.md](../experiments/exp_farras/wave5/README.md).

SHA-256 CSV utama:

```text
cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8
```

## Untuk harness tim

Gunakan [artifacts/farras_wave5.csv](artifacts/farras_wave5.csv) sebagai prediksi BAR-ATH: kolom `frame,label`, 199.319 baris pada timeline tactical 25 fps. Identitas mesin-baca ada di [method.json](method.json); rincian konfigurasi dan hasil ada di `experiments/exp_farras/`.

Jangan mengadaptasikan file BAR-ATH ini menjadi prediksi match eksternal dan jangan memakai adapter wave2 sebagai wakil wave5. Paket ini menyediakan exact target CSV untuk harness dan source untuk inspeksi; bukan API inference match baru atau bukti validasi blind.

## Metode dan alasan pemilihan

- Campuran probabilitas: 75% short-loss PP-TCN wave5 dan 25% hybrid GRU wave3 yang frozen. Short live spells lengkap di bawah 4 detik mendapat bobot state-loss dua kali; crop/unknown fragments tidak dianggap short spells lengkap.
- Decoder `dead45`: dead threshold 0.45, minimum dead 1 detik, reset H/A DP pada setiap live interval, kalibrasi jumlah possessions setelah merge.
- Gate 0.35 m, kappa 0.05, tau 3, merge 1 detik, rate scale 1. Tanpa dead-prior shift atau aturan kickoff khusus target.
- Wave5 memenuhi aturan adopsi eksternal yang ditetapkan sebelum training. Public score bukan dasar pemilihan. Wave6 memiliki public score lebih tinggi, tetapi gain development tidak mencapai batas promosi; tetap riset pembanding.

| Evaluasi yang sudah tercatat | Nilai | Bukti |
|---|---:|---|
| Development spellF1, equal mean provider | 0.670099 | [Verifikasi wave5](../experiments/exp_farras/results/wave5_verification.json) |
| Gain development terhadap wave4 | +0.013609 | Verifikasi wave5 |
| Public diagnostic spellF1 | 0.772414 | [Audit publik wave5](../experiments/exp_farras/results/wave5_public.json) |
| Public diagnostic macroF1 | 0.902059 | Audit publik wave5 |

Skor di atas adalah rekaman campaign, bukan evaluasi publik baru. Private/full-match score belum diketahui. Development, public diagnostic, dan benchmark riset IDSSE memakai protokol berbeda; angkanya bukan satu leaderboard.

## Status penyerahan

**Metode utama sudah ditetapkan; paket belum siap dikirim sebagai submission resmi.** Fresh clone dapat mengekspor dan memeriksa CSV yang dipublikasikan, tetapi itu bukan reproduksi inference dari raw bundle. Jalur raw bundle resmi ke CSV yang sama belum tersedia end-to-end di branch ini. Cache target serta model/prasyarat wave3/4/5 masih diperlukan untuk reproduksi campaign.

Laporan teknis final harus mengikuti metode ini, bukan writeup lama Ryu atau kandidat public tertinggi. Bahan sumbernya: [hasil wave5](../experiments/exp_farras/wave5/RESULTS.md), [rencana sebelum training](../experiments/exp_farras/wave5/PLAN.md), dan [atribusi data/kode](../experiments/exp_farras/wave4/reference/SOURCE.md). Dokumen ini adalah status metode, bukan laporan teknis final.

Paired ball-drop stress sedikit memburuk dan undercount belum selesai. External matches sudah terekspos historis; belum ada bukti bebas overfit atau keunggulan pada private GT. Histori dan hasil negatif tetap di [experiments/exp_farras/](../experiments/exp_farras/README.md).

## Pembagian folder

- `submission/`: pilihan utama, ekspor CSV terverifikasi, dan status kesiapan.
- `experiments/exp_farras/wave1/` sampai `wave6/`: campaign historis, tanpa pemindahan path atau perubahan source frozen.
- Campaign `fair_idsse`, `cross_branch`, `temporal_methods`, `spell_training`, `event_learning`, dan `provider_generalization`: benchmark/riset lanjutan. Kontrol atau primary dalam laporan campaign hanya berlaku pada protokol riset masing-masing, bukan pengganti metode submission.
- `experiments/exp_farras/results/`: bukti agregat dan hash. Raw data, cache, model, serta label/frame-level arrays tetap diabaikan Git. Satu pengecualian publikasi adalah CSV prediksi terpilih di `submission/artifacts/`.

Source runtime dan CSV tetap di branch `farras`. Dokumentasi keputusan dan fakta laporan diselaraskan ke `main`, `farras`, `ryu`, `jeremy`, `rasya`, dan `consolidation`; kode riset anggota tidak diganti. Perintah ekspor tidak menjalankan commit/push otomatis.

## Verifikasi penetapan

Jalur `submission/export.py` sudah dijalankan. Validator resmi menerima 199.319 baris dengan frame 0..199318; hash hasil ekspor identik dengan kandidat wave5. Pemeriksaan juga mencakup ekspor berulang yang idempotent, penolakan menimpa file berbeda, error saat CSV lokal tidak tersedia, serta penolakan kandidat yang hash-nya berubah. Sebanyak 61 tautan lokal di dokumen indeks dan petunjuk juga lolos pemeriksaan.

Rekaman terstruktur: [verification.json](verification.json). Verifikasi ini tidak menjalankan training atau scoring publik/private dan tidak membuktikan reproduksi raw bundle.

## Verifikasi paket harness

Ekspor kandidat publik juga lulus pada snapshot minimum yang hanya berisi file runtime yang dipublikasikan, tanpa raw data, tracking cache, weights atau GT. Validator resmi menerima 199.319 baris dan hash hasilnya identik dengan wave5 frozen. [Bukti publikasi](publication_verification.json). Ini pemeriksaan akses/format artefak, bukan scoring baru atau bukti inference raw bundle.
