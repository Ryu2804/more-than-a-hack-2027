# Metode final tim: Farras wave5

Tim memilih **Farras wave5 `short_gru25/dead45`** untuk kandidat submission dan isi metode laporan teknis. Pemilihan metode sudah final; kesiapan paket submission belum lengkap. Dokumen ini dan [final_method.json](final_method.json) menjadi rujukan laporan pada seluruh branch. Keputusan primary/control di laporan eksperimen lama hanya berlaku untuk campaign tersebut.

## Identitas yang dikunci

- Method ID: `farras_wave5_short_gru25_dead45`.
- Release source: commit [`7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60), branch asal `farras`.
- [CSV prediksi algoritme](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/artifacts/farras_wave5.csv): `submission/artifacts/farras_wave5.csv`, 199.319 baris, `frame,label`, timeline tactical BAR-ATH 25 fps.
- SHA-256 CSV: `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.
- SHA-256 seleksi campaign: `6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`.

CSV ini bukan GT, bukan wave6, bukan HGB wave2 yang pernah masuk tabel consolidation. Kode penelitian pada branch lain tetap milik eksperimen asal; penyelarasan dokumentasi tidak mengganti algoritmenya menjadi wave5. [Peta branch dan arsip](BRANCHES.md).

## Pipeline yang harus dijelaskan

1. Observasi berasal dari tracking otomatis target yang dipersiapkan menjadi cache posisi/jarak tim dan fitur observasi. Model bekerja offline pada 5 fps, lalu prediksi diekspansi ke timeline 25 fps. Video, human event labels target, Smart Tagging, dan homography bukan input inference pada jalur kandidat ini. Cache target belum diregenerasi end-to-end dari raw bundle oleh paket publik.
2. Komponen utama adalah ensemble PP-TCN tiga seed dengan short-possession loss wave5. State-loss diberi bobot dua kali untuk complete live spells di bawah 4 detik. Durasi dihitung pada full record per babak; fragmen crop atau unknown tidak diberi bobot short palsu.
3. Probabilitas joint H/A/D dicampur **75% short-TCN dan 25% hybrid SetSequence/GRU wave3 yang frozen**, kemudian dinormalisasi. GRU tersebut model Farras sendiri, bukan checkpoint Ryu dan bukan campuran anchor tree Metrica. GRU memakai player-set encoder dan bidirectional GRU. Jangan menyebut campuran ini boundary-head ensemble: kandidat short-TCN tidak melatih auxiliary boundary head baru.
4. Decoder `dead45` memakai dead threshold 0.45, minimum dead 1 detik, distance gate 0.35 m, kappa 0.05, tau 3, merge 1 detik, dan rate scale 1. H/A dynamic programming di-reset per live interval; kalibrasi rate menghitung possessions setelah merge. Statistik rate berasal dari training, bukan jumlah possessions GT publik.
5. Dead-prior shift dimatikan. Tidak ada forced-dead/kickoff correction khusus target pada kandidat terpilih. Batas babak resmi berasal dari metadata kompetisi, bukan possession labels buatan tangan. Fitur terpusat dan bidirectional GRU menjadikan metode offline, bukan real-time.

Konfigurasi decoder lengkap, termasuk parameter yang tidak aktif, ada di JSON. `fixed_lambda=2.0` bukan lambda efektif tetap: mode yang dipakai adalah `lambda_mode="rate"`.

## Training dan pemilihan

Komponen short-TCN dilatih pada 19 pertandingan: 5 IDSSE dan 14 SkillCorner. Development memakai enam grouped folds, seluruh corruption rounds pertandingan keluar bersama. Normalisasi dan statistik decoder memakai training fold saja. Recipe short-TCN: seeds 0/1/2, 1.500 optimizer updates per seed, batch 16, crop 384 sampel, AdamW learning rate 0.002, weight decay 0.01, OneCycle, dropout 0.15. Arsitektur memakai 64 channels, residual temporal convolutions kernel 5 dan dilations 1/2/4/8/16/1.

GRU frozen berasal dari wave3, dilatih pada lima IDSSE development matches, bukan seluruh 19 pertandingan PP-TCN. Konfigurasinya seed 2027, 8 epochs, window 300, batch 16, hidden/width 32 dan dua layer bidirectional GRU. Jangan menyatakan seluruh komponen punya training set atau jumlah seed yang sama.

Metrica Game1/2 clean/degraded digunakan untuk external development/seleksi, Game3 untuk evaluasi; bukan training neural komponen terpilih. Objective development adalah equal mean tiga provider setelah agregasi kondisi per pertandingan. Dari 96 konfigurasi, pengganti wave4 wajib naik minimal 0.01, menang minimal dua dari tiga provider, dan tanpa provider turun lebih dari 0.03. Wave5 lolos aturan tersebut sebelum public diagnostic. Angka 63 seed fits adalah total campaign tiga keluarga model dan folds/final, bukan jumlah model dalam inference kandidat final.

Data eksternal dan sumber label:

- IDSSE: dataset Bassek et al., CC BY 4.0, [paper](https://doi.org/10.1038/s41597-025-04505-y), [data](https://doi.org/10.6084/m9.figshare.28196177). Label possession dari dataset provider.
- SkillCorner [opendata](https://github.com/SkillCorner/opendata), MIT; phases-of-play provider dipakai sebagai external proxy labels.
- Metrica [sample-data](https://github.com/metrica-sports/sample-data): event-derived proxy labels untuk validasi. Audit sumber tidak menemukan lisensi eksplisit; jangan menyebut seluruh penggunaan data sudah license-clean. Identitas annotator individual tidak tercatat dalam bukti repo dan tidak boleh dikarang.

## Angka yang boleh masuk laporan

| Evaluasi | Metrik | Nilai | Makna |
|---|---|---:|---|
| Development eksternal wave5 | Equal-provider spellF1 | 0.670099305497763 | Objective seleksi, bukan skor kompetisi |
| Development versus kontrol wave4 | Delta spellF1 | +0.013608981205591952 | Perbandingan protokol sama |
| GT publik kompetisi | SpellF1 | 0.7724137931034483 | Diagnostic 5 menit, bukan leaderboard/private |
| GT publik kompetisi | MacroF1 | 0.9020586271571484 | Metrik sekunder |
| Full-match/private kompetisi | SpellF1 dan macroF1 | Belum diketahui | Tidak ada hasil private model |

SpellF1 resmi adalah mean one-to-one temporal possession F1 pada IoU 0.3/0.4/0.5/0.6/0.7, termasuk semua H/A spells pendek. MacroF1 adalah mean frame F1 H/A/D. Jangan menamai accuracy sebagai metrik utama atau membandingkan angka benchmark IDSSE dengan public score sebagai satu ranking.

Bila laporan memerlukan baseline: smooth50 resmi mempunyai public spellF1 0.325581 dan published full-match spellF1 0.2433. Keduanya cohort berbeda. Hasil wave6 public 0.786207 adalah alternatif riset yang tidak diadopsi; gain development-nya gagal batas promosi. Angka itu bukan skor metode final ini.

## Batas klaim dan kesiapan

- Tidak ditemukan aliran GT kompetisi ke training/inference/selection pada path yang diaudit. Itu bukan sertifikat leakage-free atau jaminan seluruh riwayat eksperimen bebas public-informed adaptation.
- Public diagnostics campaign sebelumnya sudah diketahui. Semua external holdout matches sudah terekspos historis; paired stress bukan pertandingan baru. Jangan menyebut fresh blind test, unbiased estimate, atau bukti bebas overfit.
- Wave5 sedikit memburuk pada paired ball-drop stress; precision serta long-spell recall masih bermasalah. Tidak ada bukti keunggulan di seluruh domain atau exact ranking final seluruh branch.
- Refit komponen baru pada prepared inputs pernah menghasilkan target CSV identik. Itu bukan raw-bundle reproduction: frozen GRU priors dan cache target masih dipakai. Hash mengunci identitas cache, bukan membuktikan asal raw-nya.
- Paket publik dapat mengekspor CSV frozen tanpa data/cache/weights/GT; raw bundle resmi ke CSV identik belum lengkap. **Jangan menulis paket sudah memenuhi seluruh syarat reproduksi RULES.md bagian 3 atau siap submit resmi.** Pemilihan metode dan kepatuhan paket adalah status berbeda.

Pada checkout release Farras, `python -B submission/export.py` memvalidasi dan menyalin CSV yang sudah dipublikasikan. Pada branch penelitian lain, jangan menganggap perintah tersebut tersedia: buka release source yang dikunci di atas. Tidak ada training atau scoring baru untuk penyelarasan dokumentasi ini.

## Sumber bukti untuk penulis Typst

Baca [final_method.json](final_method.json) sebagai data, bukan menyusun ulang angka dari chat atau mengambil maksimum seluruh laporan. Bila file `.typ` berada di `docs/`, data dapat dimuat dengan `#let method = json("final_method.json")`. Nilai private yang `null` harus ditulis belum diketahui, bukan diubah menjadi nol.

Rujukan kode dan laporan pada release yang sama:

- [Manifest runtime](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/submission/method.json).
- [Recipe dan aturan seleksi wave5](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/PLAN.md).
- [Hasil lengkap dan hasil negatif wave5](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/RESULTS.md).
- [Bukti seleksi/refit](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/results/wave5_verification.json), [audit publik](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/results/wave5_public.json).
- [Konfigurasi PP-TCN](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/models.py), [short-loss training](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/training_models.py), [GRU](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave3/models.py).
- [Pemisahan neural prior dari anchor](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/prepare_prior.py), [fusi](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/adaptive_fusion.py), [decoder](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave5/segment_decoder.py).
- [Atribusi data/kode](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/wave4/reference/SOURCE.md), [audit paparan](https://github.com/Ryu2804/more-than-a-hack-2027/blob/7254990d82ce5dec3bf1fff6baa4fe2f639e3a60/experiments/exp_farras/provider_generalization/EXPOSURE.md).

Sumber Ryu memberi fondasi PP-TCN/fitur; sumber Jeremy memberi referensi player-set/sequence model; integrasi short-loss, campuran dan decoder terpilih berada pada campaign Farras. Jangan menyalin technical writeup lama Ryu/Rasya sebagai deskripsi implementasi final. Perubahan dokumentasi ini mempertahankan source dan hasil eksperimen anggota, bukan mengklaim seluruh metode dibuat oleh satu orang.
