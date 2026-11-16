# Wave6: conservative fusion dan occlusion-aware decoding

## Dasar keputusan

Audit fetched refs mencakup main, farras, jeremy, rasya, ryu dan consolidation. Commit dan bukti rinci dicatat pada BRANCH_REVIEW.md. Consolidation membandingkan adapter berbeda, masih wave2 untuk Farras, sehingga bukan ranking model terbaru. Rasya mempunyai transfer speed-unit bug dan orientasi berdasarkan held-out GT; tidak diimpor. Ryu mempunyai nearest-opponent feature defect untuk opposite-view players; memperbaikinya perlu retraining dan tidak boleh diterapkan sepihak pada checkpoint lama. Jeremy exp7 memperbaiki damage simulasi tetapi gagal adoption intersection. Tidak menggabungkan recovered-ball detector yang gagal independent gate.

Wave5 short-loss memberi external mean spellF1 0.670099305497763, tetapi paired missing-ball stress menambah false possessions dan menurunkan long-spell recall. Hypothesis wave6: conservative mixtures, pemisahan dead/team heads, dan decoding yang mengenali visibility dapat menahan false transitions tanpa membuang short GT. Tidak ada perubahan fitur learned, training baru atau label definition.

## Grid tetap: 36 alternatif + satu kontrol

Semua bundle berasal dari probabilitas frozen: short_gru25, old_gru25, boundary_gru25; masing-masing sudah 75%TCN +25%GRU. Cached development TCN wajib OOF, bukan final-model prediction pada training match. Tidak ada target GT, provider-dependent switch atau cutoff short GT.

1. m01_blends: joint mixture short dengan old atau boundary, bobot other 0.10/0.25/0.50.
2. m02_visibility: bobot short = missing +(visible-missing)*mean11(visibility), visible 0.85/1.0 dan missing 0.25/0.50/0.65; sisanya old. Centered window dipotong pada babak.
3. m03_heads: conditional-team short tetap; dead probability short dicampur old/boundary dengan bobot other 0.25/0.50/0.75. Reconstruct joint probabilities.
4. m04_occlusion_cost: short tetap, switch penalty multiplier 1+(factor-1)*(1-mean_window(visibility)), factor 1.25/1.5/2.0, window1/11. Cost untuk transition masuk frame, tidak melintasi dead/half reset.
5. m05_dead_hysteresis: connected p_dead>low run, low0.35/0.40, high0.50/0.55/0.60. Run minimal5samples dan mengandung p_dead>high; terpisah setiap babak.
6. m06_short_guard: hanya low-confidence occluded A-B-A excursion yang boleh digabung. B berdurasi <=1/2/3detik, mean ball-visible<0.5, mean conditional confidence B<0.60/0.70; kedua tetangga A harus sama, live, dalam babak sama. Putusan simultaneous dari original runs, tidak cascading. Strong short possessions tetap dipertahankan.

Kontrol: exact frozen wave5 short_gru25/dead45. Default decoder dead threshold0.45/min1s, H/A live-reset DP, gate0.35m,kappa0.05,tau3, merge1s. Seluruh18 lambda diperiksa; target count dari training fold; kalibrasi menggunakan jumlah possessions setelah merge DAN custom postprocess. Tidak mengasumsikan count monoton. Tidak ada prior shift, kickoff khusus target, rate tuning atau 36-way public sweep.

## Data dan objective

Reused six grouped folds:19 training matches (5 IDSSE,14 SkillCorner), normalisasi/probabilitas/rate asli fold. Development23 scenarios/21 unique matches, Metrica1/2 clean+r0 hanya evaluasi zero-shot. Kondisi dalam match dirata-rata lebih dahulu, kemudian mean match per provider, kemudian equal mean tiga provider. Primary spellF1 resmi semua H/A spells; macroF1 hanya tie-breaker.

Rank spellF1, macroF1, identifier. Alternatif eligible bila gain >=0.005 dari wave5, menang minimal2/3 provider, tidak ada provider turun>0.01, dan lower paired match-bootstrap95% >0. Bootstrap5000,seed20261114, resample unique matches WITHIN provider, seluruh conditions satu match ikut bersama. Interval percentile tidak dikoreksi multiple comparisons dan memakai development adaptif; bukan bukti konfirmatori atau probabilitas private win. Jika tidak eligible, kontrol tetap terpilih. Best raw alternative tetap dilaporkan, tidak dihapus.

## Exposure dan tahap evaluasi

Seluruh data lama, public score sebelumnya, dan paired stress wave5 sudah terekspos. Ini eksplorasi adaptif, bukan fresh blind holdout. Tidak ada klaim prestasi private/full-match. Stress6 berasal dari6 SkillCorner holdout yang sama, bukan6 match tambahan. Holdout biasa10scenarios/9 unique matches; total dengan stress16scenarios, masih9 unique matches.

Source/PLAN, cached probabilities, old manifests, data dan external-source digests dibekukan sebelum evaluasi baru. Selection dibekukan sebelum wave6 holdout/public. Tiga role fixed: selected,best_new,previous_wave5; duplicate roles bukan model independen. Tidak memilih ulang berdasarkan holdout, stress atau public macroF1.

Independent process rebuild memuat ulang frozen probability caches dan mengulang decoding sampai target CSV identik199319baris. Ini bukan model refit; bukti refit learned components diwarisi wave5/wave3. Pipeline inference yang membutuhkan cache hilang harus gagal, bukan diam-diam menulis atau meregenerasi hasil campaign lama.

## Pelaksanaan dan batas publikasi

Enam implementation slices, parent memegang protokol/runner/seleksi. Actual evaluation satu proses CPU karena RAM tersedia sekitar1.57GB; tidak menjalankan enam proses Torch. Tidak ada training seed fits baru. Semua metode memakai source API yang sama, tidak menyalin seluruh pipeline.

Regression checks hanya untuk edge nyata: normalization/conditional heads, half-local visibility, hysteresis seed/min-duration/half boundary, transition-cost semantics, simultaneous A-B-A corrections, high-confidence short preservation, postprocess count calibration. Jalankan sekali setelah integrasi, sebelum freeze. Jika gagal, perbaiki sebelum freeze lalu jalankan check yang gagal.

Raw tracking, labels/probabilities per frame, weights dan submission CSV tetap ignored dalam cache/out. Kode, audit branch, dan hasil agregat boleh dibagikan. Tidak ada commit/push otomatis; source teman dan wave1..5 tidak dimodifikasi.
