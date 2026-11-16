# Wave5: batas possessions, short spells dan missing-ball robustness

## Pertanyaan

Wave4 meningkatkan public spellF1, tetapi short possessions masih terlewat, worst development count error44,62%, dan Metrica3 lebih buruk daripada wave3. Tidak menyimpulkan bahwa campuran GRU diperlukan hanya karena skor publik sama dengan PP-TCN. Campaign ini menguji intervensi training terpisah, reset tim saat dead-ball, serta kalibrasi rate setelah seluruh postprocessing.

Semua kode/input wave1/2/3/4 read-only. Target GT tidak dipakai training, ranking, atau memilih hyperparameter. Skor publik wave4 sudah diketahui; wave5 adalah riset adaptif, bukan test independen baru.

## Data dan pemisahan

- Training19 pertandingan licensed dari wave4:5 IDSSE development,14 SkillCorner development. IDSSE memakai3 corruption rounds yang sama. Tidak menambah Metrica training.
- Enam grouped folds wave4 tetap: pertandingan sorted di tiap provider diberi index modulo6, seluruh round pertandingan keluar bersama. Enam fold training15/15/16/16/16/17 match. Normalisasi dan rate statistik hanya dari training fold.
- Development23 scenarios/21 match:IDSSE5 r0,SkillCorner14 clean,Metrica1/2 clean/degraded. Metrica untuk zero-shot stress, tidak melatih model baru/GRU prior.
- Objective per match mean spellF1, lalu mean per provider, lalu equal mean IDSSE/SkillCorner/Metrica. MacroF1 tie-breaker, lalu family/decoder lexicographic. Jangan membuang short GT.
- Holdout lama:IDSSE2 r0,SkillCorner6 clean,Metrica3 clean/degraded. Semua pernah dievaluasi; jangan disebut blind.
- Setelah selection freeze, buat pasangan stress untuk6 SkillCorner holdout. Seed20261113 ditambah identifier match stabil; drop tambahan35% baris bola yang semula terlihat dalam contiguous spans10..60 sampel, terpisah per babak. Pemain, label dan timeline tidak diubah. Seluruh X16 dan fitur PP/ball dihitung ulang dari input yang dimask. Stress adalah uji transformasi baru pada match lama, bukan6 pertandingan independen baru.

## Training tiga model

Basis PP-TCN wave4 tetap:64 channels,6 residual Conv1d kernel5/dilations1,2,4,8,16,1; GroupNorm8, dropout0.15, crop384,batch16,1500optimizer steps,AdamW lr0.002/weight_decay0.01,OneCycle. Seeds0/1/2, provider/match/round balanced sampling. Baseline mask_full0.3,pooled-player noise0.5; seluruh kanal PP yang memakai bola dimask konsisten. Full-block two-view inference; CPU checkpoint nets, actual CUDA fit setelah smoke lolos.

1. `boundary_tcn`: tambah output head boundary; dua state losses tetap, ditambah0.2 BCE boundary. Positif bila ada pergantian known state adjacent, dilatasi +/-2frames dalam label known, fixed positive weight5. First crop frame dan unknown-adjacent changes diabaikan.
2. `short_tcn`: dua state heads tetap. Faktor2 pada state-loss untuk live GT spells <20frames/4s. Durasi diukur pada full record per babak, bukan crop; run yang terpotong unknown tidak dihitung sebagai short lengkap. Tidak sekaligus mengubah sampler.
3. `masked_tcn`: dua state heads tetap; full ball mask probability0.6, bukan0.3. Noise dan recipe lain tetap.

Tiap model enamfold plus final19-match fit, ensemble3seed:21 ensemble fits,63 seed fits dengan recipe penuh. Checkpoint PP-TCN wave4 hanya final; cached dev PP predictions adalah OOF asli. Tidak memakai final checkpoint untuk memalsukan prediksi SC development heldout. Tidak memakai shipped Ryu weights untuk IDSSE/SkillCorner OOF.

## Dua belas probability families

`old_pp`, `old_gru25`, `boundary_tcn`, `short_tcn`, `masked_tcn`, `boundary_gru25`, `short_gru25`, `masked_gru25`, `quality_old`, `new_ensemble`, `new_ensemble_gru25`, `quality_new`.

- Old PP dari cached OOF wave4 pada development, final checkpoint zero-shot pada holdout. GRU dari OOF IDSSE/zero-shot SC/Metrica prior wave3 yang sudah dihash wave4. Tidak memakai anchor tree.
- `old_gru25` persis75% PP wave4 +25% GRU hybrid lama. Ini kontrol frozen wave4.
- Tiga `_gru25` baru memakai75% model primer +25% GRU.
- `new_ensemble` mean joint probabilities ketiga model baru; boundary hanya head yang benar-benar dilatih dari `boundary_tcn`, tidak merata-ratakan placeholder.
- Quality family: primer oldPP atau newensemble; GRU weight `.05+.35*visibility*entropy(primary conditional-team)/log(2)`. Visibility adalah centered11sample mean X[:,3] dalam babak. Weight di [.05,.4]. Aturan fixed, tidak belajar kalibrasi dari GT atau provider identity.
- Semua campuran joint probability dinormalisasi lalu conditional dead/team dihitung ulang. Boundary primer dipertahankan; oldPP tanpa learned boundary diberi provenance false.

## Delapan decoder

Semua tanpa dead-prior shift. Default gate0.35m,kappa0.05,tau3,merge1s,dead threshold0.5,min dead2s,rate_scale1. Rate target dari provider-balanced training spells/menit. Grid lambda asli wave4 dipakai lengkap; nearest count/tie lambda terkecil, tidak mengasumsikan monotonic.

1. `wave4`: persis decoder `no_shift` lama; rate calibration sebelum merge, intentional control.
2. `postmerge`: H/A DP whole-half sama, tetapi calibration menghitung final labels setelah merge.
3. `live_reset`: postmerge plus H/A DP independen di tiap live interval, tidak membawa team state melintasi dead.
4. `boundary_reset`: live_reset, switch cost per frame `lambda*(1-.5*boundary_score)`. Pakai learned head bila tersedia; selain itu `clip(abs(diff(logit(p_team)))/4,0,1)` dengan reset pada awal half/live interval. Jangan menganggap placeholder0.5 sebagai learned head.
5. `dead1`: live_reset,min dead1s.
6. `dead45`: live_reset,threshold0.45,min dead1s.
7. `dead55`: live_reset,threshold0.55,min dead2s.
8. `no_merge`: live_reset,merge0.

12x8=96 konfigurasi,2.208 development evaluations. Count error dan short/medium/long diagnostics dilaporkan tetapi tidak mengganti metric utama.

## Pemilihan dan keputusan adopsi sebelum hasil

Kontrol frozen adalah `old_gru25/wave4`. Ranking penuh tetap objective di atas. Konfigurasi baru eligible untuk mengganti kontrol hanya jika mean spellF1 sekurangnya0.01 di atas kontrol, menang pada sekurangnya2 dari3 provider means, dan tidak ada provider turun lebih dari0.03. Pilih eligible terbaik; bila tidak ada, pertahankan kontrol. Ini guard terhadap kenaikan kecil pada development yang berulang kali dipakai. Tetap simpan `best_new` menurut ranking di luar kombinasi kontrol agar hasil negatif transparan.

Sebelum public, hash source/input/recipe/device, freeze selection/ranking/checkpoint hashes. Sesudah itu hanya kandidat fixed: `selected`, `best_new`, `previous_wave4`, `pure_boundary` (`boundary_tcn/wave4`). Role yang identik bukan model independen. Holdout16 scenarios termasuk6 paired stress. Diagnostik per durasi <4s,4-15s,>=15s dengan one-to-one IoU matching scorer resmi; tidak membuang short GT dari denominator.

Independent refit melatih ulang semua komponen baru yang diperlukan oleh kandidat selected dari seed yang sama; frozen komponen lama hanya diverifikasi checksum dan bukti refit lama. Jika selected tidak memakai komponen baru, jangan mengaku ada new-component refit. CSV199.319baris harus identik setelah refit/rebuild. Public scorer resmi baru dipanggil setelah tahap ini, tanpa mengganti pilihan.

## Artefak dan batas

Kode/reports di `experiments/exp_farras/wave5/`, agregat di sibling `results/wave5_*.json`. Raw data, prepared packs, features, probabilities, checkpoints dan submission CSV tetap ignored. Tidak mengubah source atau artifact wave lama. Tidak ada commit/push otomatis atau upload tracking.

SkillCorner MIT dan IDSSE CC BY4.0, kredit diwarisi dari wave4/reference/SOURCE.md. Metrica hanya evaluasi. Statistik diperlakukan sebagai proxy dan riset adaptif; tidak ada klaim private/full-match, significance, atau fresh-match holdout yang tidak tersedia.
