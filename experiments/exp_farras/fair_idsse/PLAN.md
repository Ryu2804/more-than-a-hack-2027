# Fair IDSSE benchmark: protocol v1

Tujuan: mengganti ranking consolidation yang mencampur adapter, training membership dan exposure GT. Ini benchmark keluarga model yang diadaptasi pada data/protokol sama, bukan reproduksi exact submission tim atau ranking final branch.

## Data, split, labels

Tujuh IDSSE matches J03WMX,J03WN1,J03WOH,J03WOY,J03WPY,J03WQQ,J03WR9. Pakai prepared wave3 r0 yang sama byte-for-byte untuk semua arms, corruption seed20261108+index match. Tidak memakai r1/r2, SkillCorner, Metrica, target BAR-ATH, GT publik atau checkpoint lama. Team mapping dari match metadata home/away, bukan agreement dengan GT. Degrader menerima tracking/team/half/seed, bukan labels. Provider labels H/A/D asli, unknown -1, tanpa merging/filter short GT.

Leave-one-match-out7 folds. Seluruh arms terlatih wajib memakai persis6match lain, bukan2match atau final weights yang pernah melihat test. Unknown training labels diabaikan; unknown test labels hanya dipakai scorer, tidak inference. Normalisasi/rate/dead_share hanya dari6training matches. Half-local5fps dipetakan ke25fps memakai blocks25 identik untuk semua output. Runner memeriksa timeline dan ukuran prediction; tidak memberi hidden labels/X mask/orientasi GT kepada predictor.

Semua match sudah terekspos riset tim. LOSO ini memberi match-exclusive fits baru, bukan fresh blind test atau bukti keunggulan private.

## Arms dan budget fixed

1. HGB symmetric: keluarga wave2 Farras, seluruh6match masuk training pool; deterministic balanced sample hingga40000rows sesudah H/A augmentation,100iterations,15leaves,lr.08,l2=1. Tidak memakai truncation first2matches. FeatureX16 sama asalnya dengan semua model.
2. SetSeq: keluarga DeepSets+BiGRU Jeremy, implementasi adapted wave3 setseq (bukan exact exp7). Width/hidden32, state CE +boundary BCE0.5,pos_weight3, mirror augmentation, AdamW lr.002 wd.0001,clip1.
3. PP-TCN: keluarga Ryu adapted wave4, width64,dilations1/2/4/8/16/1,mask_full.3,noise.5,AdamW lr.002 wd.01,OneCycle.
4. Short-loss PP-TCN: keluarga Farras wave5; dua head, faktor2loss complete live spells<4s; recipe lain sama PP-TCN.
5. Touch-hold: keluarga Rasya adapted5fps, radius.8m,min carriedrun1.2s; dead cue out-of-pitch>=.6s atau nearest-player speed<2m/s>=1s, visible ball/valid metadata required. Speed dari posisi meter/elapsedseconds, bukan /3.6. Half reset, leading pre-touch D, fixed dead merge0. Tidak memakai GT flip, GT dead mask atau GT X mask. Ini aturan label-free yang berbeda dari original oracle-dependent implementation.

Semua neural arms3seeds0/1/2,500optimizer updates/seed,batch16,window/crop384. SetSeq explicit fixed-step training, bukan epoch budget tersembunyi. Total63fresh seed fits/31500updates; HGB7fits,rule0. Budget ini fixed sebelum hasil diketahui, lebih kecil daripada recipe produksi1500steps; interpretasi adalah equal-update-budget benchmark. Equal steps bukan equal FLOPs/walltime. Optimizer/loss/augmentation dan representasi model tetap berbeda dan dilaporkan; tidak mengklaim causal ablation arsitektur saja. Tidak ada tuning atau memilih epoch berdasarkan outer fold.

## Decoder dan metric

Semua learned joint probabilities diberi decoder identik frozen wave5 dead45: threshold.45,minD1s,live-reset H/A DP,gate.35m,kappa.05,tau3,merge1s,full18lambda grid nearest fold-training rate sesudah merge. Learned boundary heads tidak dipakai decoder, termasuk SetSeq. Argmax tiap learned arm dilaporkan terpisah sebagai diagnostic yang sama. Rule hard labels ditaruh panel native-rule; bukan mengklaim mendapat decoder sama. Hindari perbandingan branch-specific tuned decoder di tabel utama.

Metric resmi mean spellF1 IoU.3/.4/.5/.6/.7, semua short GT; macroF1 secondary. Mean unweighted7unique heldoutmatches. Tidak ada pemilihan kandidat submission atau promosi berdasarkan tabel. Paired bootstrap5000(seed20261114) atas match, bukan frame/seed;95% percentile exploratory, tidak familywise corrected. Semua pair methods dilaporkan supaya tidak hanya menonjolkan pemenang.

## Integrity, verification, limits

PLAN/source/dependency files, IDSSE provenance audit, data packs dan feature packs dihash sebelum fits. Setiap job cache terikat manifest hash, exact training names, heldout name,family,seeds,steps; mismatch gagal bukan reuse. API predict menolak labels dan overlapping training/test names. Old weights tidak pernah dibuka. Tidak ada fallback memperbaiki hasil lama.

Koreksi audit sebelumnya: file referensi Ryu memang memakai opponent mask relatif view, tetapi feature builder aktual kita `wave4/features.py:_player_set` memakai `player_team[j] != player_team[k]`, bukan mask view. Menganggap bug salinan referensi otomatis diwarisi model kita tidak tepat. Benchmark memakai feature packs builder aktual yang sama untuk kedua PP arms, bukan menyalin reference builder. Default decoder sebelumnya dipilih dari data terekspos, walaupun tidak dituning di sini. Semua arms offline full-half, bukan streaming.

Sebelum freeze jalankan semantic tests, actual reduced1step fit/predict tiap neural pada slice data nyata, HGBfit/predict dan actual rule. Full benchmark harus selesai semua folds; laporan tidak boleh campur partial/old scores. Independent fresh-process rebuild decoding dari output caches untuk semua folds/arms diverifikasi hash labels. Tidak menyebutnya model refit.

Raw tracking, labels/probabilities per frame, checkpoints dan out/cache ignored. Hanya kode/protokol serta metrik/provenance agregat dibagikan. Tidak ada commit/push; source tim dan wave1..6 immutable. IDSSE CC BY4.0; provenance/source credits diwarisi wave3/reference/SOURCE.md dan wave4/reference/SOURCE.md.

## Runtime correction sebelum full preregistration

32 semantic tests lulus. Smoke gabungan menyelesaikan seluruh fit/predict dan menulis JSON, tetapi proses Windows keluar fatal dengan kode -1073740791. Hasil itu tidak diterima sebagai runtime pass. Sebelum full run, tiap family/fold dipindahkan ke child process tersendiri, parent hanya scoring. Child harus exit0 sebelum cache dibaca; error tidak ditelan. Budget/data/model tidak berubah. Smoke baru memverifikasi tiap proses secara terpisah sebelum registry full fits dibuat.

Sampling juga tetap bagian recipe: SetSeq mengambil windows proporsional jumlah valid starts, PP-TCN menyeimbangkan match, HGB mengambil balanced sample dari global pool. Seluruhnya boleh mengakses pool6match yang sama, tetapi distribusi sampling/loss tidak dipaksakan identik. Hasil membandingkan fixed adapted pipelines dalam common evaluation, bukan eksperimen causal yang hanya mengganti arsitektur.

SetSeq-only CUDA child juga selesai menghitung tetapi gagal exit dengan status3221226505; isolasi saja belum cukup. Backend sebelum freeze diubah: SetSeq CPU, dua PP-TCN CUDA; HGB/rule CPU. Jumlah updates/seeds/data tetap. Registry mengikat backend per family, worker memverifikasinya. Karena hardware berbeda, hasil tidak dipakai ranking latency/efisiensi. Acceptance tetap requires exit0, bukan sekadar adanya JSON.
