# Hasil eksperimen Farras

## Penetapan metode final tim

Tim memakai **wave5 `short_gru25/dead45`** sebagai metode final. Laporan teknis mengikuti [spesifikasi lintas branch](../../docs/FINAL_METHOD.md) dan [data fakta Typst](../../docs/final_method.json). Penetapan ini memakai seleksi frozen yang sudah lolos aturan adopsi eksternal, bukan evaluasi publik atau training baru. [Identitas/hash kandidat](../../submission/method.json), [ekspor dan status kesiapan](../../submission/README.md), [hasil wave5](wave5/RESULTS.md).

Wave6 dan campaign berikutnya tetap sebagai riset pembanding. Keputusan primary/kontrol di tiap benchmark hanya berlaku pada protokolnya sendiri. Histori di bawah tidak ditulis ulang. Paket belum siap untuk penyerahan resmi karena reproduksi raw bundle ke CSV yang sama belum lengkap.

Untuk harness tim, satu CSV prediksi terpilih diterbitkan di `submission/artifacts/farras_wave5.csv`. Hash-nya tetap identik dengan kandidat wave5 frozen. Publikasi source/bukti campaign tidak melatih ulang model atau menjalankan scoring baru; cache, weights, raw data dan GT tidak ikut.

## Riset lanjutan

Generalisasi terbaru: [provider_generalization/RESULTS.md](provider_generalization/RESULTS.md). Source-provider-only training dua arah tanpa competition GT, 30 seed fits, lima resep. Kontrol equal-provider spellF1 **0.540864**; RDrop **0.540882**, selisih praktis nol dengan interval melintasi 0. Primary modality dropout **0.530378** gagal promosi. Kontrol dipertahankan. Sebanyak 58 tests lulus/1 skip, 300 artifacts tersegel sebelum scoring, 270 probability/decoded arrays dan metric rows direproduksi identik. **Tidak menjamin bebas overfit:** semua 27 external matches sudah terekspos historis; belum ada fresh blind cohort. Tidak sebanding langsung dengan angka ensemble IDSSE lama. [Audit paparan dan read-isolation limits](provider_generalization/EXPOSURE.md).

Enam resep lanjutan: [event_learning/RESULTS.md](event_learning/RESULTS.md). Best exploratory event+contact ensemble mencapai mean spellF1 IDSSE **0.624577**, naik0.005014 dari strong research control0.619563, menang5/7; interval masih melintasi0. Primary event+balanced0.619192 gagal promosi, strong control tetap dipakai. Ranking/dead45 mencapai0.621512 dan menang7/7 versus strong consensus control; same-decoder contrast belum terpisah jelas.126fresh seed fits,77tests,42saved-model replays dan280decoded arrays identik. Tidak ada public/private score baru.

Training baru terbaru: [spell_training/RESULTS.md](spell_training/RESULTS.md). Best exploratory event ensemble + consensus mencapai spellF1 IDSSE **0.619563**. Primary event ensemble/dead45 **0.618941**, naik0.003484 dari kontrol0.615458, menang5/7 dengan interval positif tetapi belum memenuhi gain minimum0.005. Kontrol tetap dipertahankan.63seedfits baru,27tests lulus,21saved-model inference replays dan140decoded arrays identik. Standalone event TCN juga membaik0.608023 versus short-TCN lama0.595755; tidak ada public/private score baru.

Metode baru terbaru: [temporal_methods/RESULTS.md](temporal_methods/RESULTS.md). Best exploratory consensus mencapai spellF1 IDSSE **0.616970**, naik0.001512 dari kontrol0.615458 tetapi interval selisih masih melintasi0. Primary HSMM0.605838 gagal promosi; kontrol tetap dipertahankan.19settings/133evaluasi,47tests lulus,133decoded arrays dan7refittedpriors identik. Tidak ada public/private score baru.

Adaptasi lintas branch terbaru: [cross_branch/RESULTS.md](cross_branch/RESULTS.md). Primary gabungan contact/memory/PP-TCN/short-loss/SetSeq mencapai spellF1 IDSSE **0.609030**, kontrol0.595755, menang6/7 dan lolos aturan riset. Best raw alternative short75+SetSeq25 mencapai **0.615458**, menang7/7; tidak mengganti primary post-hoc. Semua28settings dan hasil negatif tersedia; tidak ada public/private score baru.

Benchmark bersama terbaru: [fair_idsse/RESULTS.md](fair_idsse/RESULTS.md). Mean spellF1 IDSSE: short-loss PP-TCN0.595755, SetSeq0.578602, PP-TCN0.571090, HGB0.315719. Semua learned arms training ulang pada6match lain tiap fold dan memakai decoder sama. Bukan ranking exact submission atau skor publik; interval short-TCN versus SetSeq masih melintasi0.

Hasil public diagnostic terbaru tetap [wave6](wave6/RESULTS.md):0.786207 untuk best exploratory alternative, selected tetap wave5 (publik0.772414). Tidak memilih ulang dari GT publik. Audit fitur sebelumnya dikoreksi dalam laporan benchmark: reference Ryu dan builder aktual kita bukan fungsi sama. Riwayat: [wave5](wave5/RESULTS.md), [Ryu/wave4](wave4/RESULTS.md), [Jeremy/Farras/wave3](wave3/RESULTS.md). Laporan di bawah mempertahankan keputusan dan angka historis dua batch pertama.

## Riwayat wave1 dan wave2

72 konfigurasi dalam dua batch, 288 evaluasi development eksternal. Metrik utama spellF1; frame macroF1 hanya tie-breaker pemilihan. Seluruh skor publik di bawah adalah evaluasi lokal pada window GT publik yang sama, bukan hasil leaderboard private.

| Pembanding | Public spellF1 | Keterangan |
|---|---:|---|
| Baseline resmi smooth50 | 0.325581 | Pembanding publik |
| Hasil Ryu yang direview sebelum campaign | 0.528571 | Control gate; bukan klaim skor terbaru branch Ryu |
| Wave1, kandidat dipilih eksternal | 0.442105 | Hysteresis radius1.5m |
| Wave2, kandidat dipilih eksternal | 0.460606 | Augmentasi H/A + histogram + Viterbi6 |
| Wave2, diagnostic publik tertinggi | 0.560000 | 7 fitur kontrol + histogram + Viterbi6; bukan pilihan eksternal |

Baseline full/private yang dipublikasikan kompetisi adalah0.2433. Tidak dibandingkan langsung dengan angka publik di tabel; skor private eksperimen ini belum diketahui.

## Konfigurasi dengan public0.56

`wave2/methods/w14_feature_ablation.py`, konfigurasi `control`:

1. Memakai cache `atd_full.npy` dan `team_dist.npy` dari pipeline Ryu untuk target. Tidak memodifikasi pipeline asal.
2. Memakai7 fitur: jarak pemain terdekat tiap tim ke bola, margin kedua jarak, status bola terlihat, jarak bola ke batas lapangan, gap age dan gap remaining. Fitur posisi absolut, kecepatan, dan shape pemain tidak dipakai.
3. `HistGradientBoostingClassifier`: max_leaf_nodes15, max_iter100, learning_rate0.08, l2_regularization1, early_stoppingFalse, seed2027. Bobot kelas seimbang, maksimal40.000 baris training berlabel.
4. Training dari Metrica Game1/2, kondisi asli ditambah salinan oklusi sintetis. GT target tidak dipakai training.
5. Viterbi penalti pergantian6, terpisah per babak. Tidak memakai augmentasi pertukaran tim pada konfigurasi ini.
6. Prediksi5fps diekspansi ke25fps; gap remaining membuat pipeline offline, bukan real-time.

Development external mean spellF1:0.500969, macroF1:0.728420. Diagnostic publik spellF1:0.560000, macroF1:0.650520, 13 predicted possessions versus17 GT pada window publik.

Public0.56 tidak digunakan untuk mengganti kandidat frozen. Menetapkannya sebagai pemenang setelah melihat GT publik akan mencampur test diagnostic dengan pemilihan model.

## Kandidat frozen dan holdout

| Evaluasi spellF1 | Wave1 | Wave2 |
|---|---:|---:|
| Mean Game1/2, clean/occluded | 0.460517 | 0.517438 |
| Game3 kondisi asli | 0.474935 | 0.599664 |
| Game3 tambahan oklusi | 0.374534 | 0.433982 |

Wave2 frozen: `w08_symmetric_boost/cost6`. Fitur H/A ditukar beserta margin dan label pada training, kemudian histogram gradient boosting + Viterbi6. CSV target SHA-256:

`45e03662fccbeb238e9df723c99829df60e3d8d641d539be9f6d1e1fd59b0446`

Kandidat wave1 SHA-256:

`6a022089301a08d9d469fbc4630cb65c3f4a34b1d4de1193488422370e63b52d`

Game3 tidak dipakai training/tuning atau memilih pemenang. Hanya dua kandidat frozen diuji pada kedua kondisi. Parser memeriksa11 susunan channel metadata, 35 identitas pemain, serta periode event untuk menghindari kesalahan slot pemain dan babak.

## Temuan negatif dan batas

Shape-only lebih lemah daripada fitur kontrol pada protokol ini. GaussianNB, confirmation gate, learned transition logistic, dan bridging dead-ball tidak mengalahkan kandidat terpilih. Model yang lebih rumit tidak otomatis memberi spellF1 lebih tinggi.

Semua72 konfigurasi memakai dua pertandingan development yang sama; pencarian bersifat adaptif. Game3 hanya satu pertandingan holdout, labelnya proxy event dan tidak identik definisi challenge. Tambahan oklusi masih menurunkan spellF1 dengan jelas.

Skor Ryu di atas berasal dari review cache eksperimen sebelumnya. Branch remote tim berkembang setelah review; tidak ada klaim bahwa0.56 mengalahkan hasil terbaru semua anggota tim. Tidak ada perbandingan private/full-match yang setara.

## Bukti yang dipush

`results/` menyimpan skor agregat seluruh konfigurasi, konfigurasi kandidat, metrik holdout, dan hash reproduksi. `out/` lokal menyimpan CSV/prediksi per frame dan preregistrasi; tidak dipush. Raw data/cache juga tidak dipush.

Setelah pemindahan lokasi ke repo, kedua campaign dijalankan ulang dengan input yang sama. Perubahan lokasi/import tidak mengubah algoritme atau konfigurasi. Hasil reproduksi serta perbandingan dengan campaign asal dicatat di `results/relocation_verification.json`.

Perintah dan kebutuhan input: [README.md](README.md).
