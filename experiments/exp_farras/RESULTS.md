# Hasil eksperimen Farras

## Ringkasan

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
