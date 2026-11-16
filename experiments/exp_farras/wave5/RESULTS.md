# Wave5: short-possession weighting lolos validasi, publik tetap

**Kandidat frozen: `short_gru25/dead45`.** Lolos aturan adopsi yang ditulis sebelum training. Mean spellF1 eksternal naik pada protokol yang sama; spellF1 publik tidak naik. Hasil negatif pada paired ball-drop stress dipertahankan, bukan disembunyikan oleh kenaikan short-spell recall.

## Keputusan sebelum hasil publik

Primary objective tetap mean spellF1 provider-balanced, macroF1 tie-breaker. Kandidat baru harus naik minimal0.01 dari kontrol `old_gru25/wave4`, menang minimal2/3 provider, dan tidak ada provider turun lebih dari0.03. Dari96 konfigurasi, 6 memenuhi syarat. Kandidat terbaik eligible juga merupakan best raw alternative, sehingga `selected` dan `best_new` adalah role yang sama, bukan dua model independen.

| Provider development | Kontrol wave4 | Selected wave5 | Selisih spellF1 |
|---|---:|---:|---:|
| idsse | 0.583755 | 0.612109 | +0.028354 |
| skillcorner | 0.709044 | 0.709337 | +0.000293 |
| metrica | 0.676673 | 0.688852 | +0.012179 |
| Equal mean provider | 0.656490 | 0.670099 | +0.013609 |

SkillCorner development hampir datar; jangan menyebut seluruh provider meningkat besar. Worst development absolute count error turun dari44,62% menjadi41,88%, masih melampaui30%. Tidak ada klaim masalah short-spell/undercount sudah selesai.

Kandidat terpilih memakai:

- 75% PP-TCN baru dengan faktor2 pada state-loss untuk live GT spells **<4detik**, ditambah25% GRU hybrid wave3 yang frozen.
-  Panjang short spell dihitung pada full record di dalam babak, bukan fragment crop. Run yang terpotong unknown tidak mendapat bobot short palsu.
-  Decoder `dead45`: dead threshold0.45,min dead1s; H/A DP reset di tiap live interval; target rate dikalibrasi pada **jumlah possessions setelah merge**.
-  Gate jarak0.35m,kappa0.05,tau3,merge1s,training rate scale1. Tidak memakai dead-prior shift, target GT atau aturan kickoff khusus target.

Aturan lengkap yang dihash sebelum fit: [PLAN.md](PLAN.md).

## Apa yang dicoba

Tiga model baru mempertahankan PP-TCN wave4 dan recipe1500steps/seed, batch16,crop384,AdamW lr0.002/weight_decay0.01,OneCycle,dropout0.15,seeds0/1/2. Intervensi dipisahkan:

1. `boundary_tcn`: output head ketiga, auxiliary BCE weight0.2,pos_weight5; target known-state changes dengan toleransi+/-2frames. First crop frame/unknown adjacency diabaikan.
2. `short_tcn`: double state-loss pada complete short live spells; sampler, arsitektur dua head dan masking tetap.
3. `masked_tcn`: full-ball-mask probability0.6, dibanding0.3 pada basis; noise dan parameter lain tidak diubah.

Delapan decoder: kontrol wave4, post-merge calibration, live reset, boundary-dependent transition cost, minimum dead1s, threshold0.45,threshold0.55, dan tanpa merge. Boundary-dependent DP memakai learned head bila tersedia; selain itu perubahan conditional-team logit, bukan placeholder0.5. Semua grid lambda diperiksa; tidak mengasumsikan count monoton.

Dua belas probability families mencakup kontrol PP/GRU lama, tiga model baru dan campuran25%GRU, ensemble tiga model, serta quality weighting dari visibility dan conditional entropy. Tidak ada learned calibrator, provider-label switch atau anchor tree Metrica.

| Family | SpellF1 dengan decoder wave4 | Decoder terbaik | SpellF1 terbaik |
|---|---:|---|---:|
| old_pp | 0.649605 | boundary_reset | 0.652122 |
| old_gru25 | 0.656490 | boundary_reset | 0.658683 |
| boundary_tcn | 0.646890 | dead45 | 0.650364 |
| short_tcn | 0.659208 | boundary_reset | 0.661649 |
| masked_tcn | 0.646267 | postmerge | 0.648476 |
| boundary_gru25 | 0.655136 | boundary_reset | 0.660934 |
| short_gru25 | 0.664997 | dead45 | 0.670099 |
| masked_gru25 | 0.655079 | boundary_reset | 0.658034 |
| quality_old | 0.657831 | dead45 | 0.659872 |
| new_ensemble | 0.652994 | dead45 | 0.656644 |
| new_ensemble_gru25 | 0.660147 | boundary_reset | 0.664632 |
| quality_new | 0.659846 | dead45 | 0.666705 |

Dengan decoder yang sama, short-loss model lebih baik daripada PP lama. Auxiliary boundary head dan full-ball-mask0.6 tidak mengalahkan basis PP pada kontrol matched. Ensemble semua model tidak otomatis lebih baik dari short model; menambahkan model lemah dapat meredam keuntungan. Perubahan decoder saja pada kontrol GRU lama tidak mencapai margin adopsi0.01.

Seluruh96 konfigurasi dan2.208 rows development: [wave5_development.json](../results/wave5_development.json). Angka konfigurasi bukan jumlah arsitektur atau model independen. Secara kumulatif lima campaign mencakup298 konfigurasi dan4.562 evaluasi development yang sebagian memakai match sama.

## Pemisahan data dan exposure

Training tetap19 licensed matches:5 IDSSE dan14 SkillCorner. Enam grouped folds mengeluarkan seluruh round match yang diuji; IDSSE memakai3 corruption rounds. Normalisasi/rate hanya dari training fold. Metrica1/2 clean/degraded dipakai zero-shot stress, bukan training model baru maupun GRU. Development23 scenarios berasal dari21 match; kondisi dalam match dirata-ratakan sebelum provider-balanced objective.

Cached PP development adalah OOF wave4 asli. Checkpoint PP final tidak dipakai untuk membuat prediksi palsu held-out pada SkillCorner development. Shipped Ryu weights tidak masuk wave5, karena pernah dilatih pada seluruh IDSSE/SkillCorner. Input dan kode seluruh wave lama tetap read-only.

Holdout lama mencakup2 IDSSE,6 SkillCorner dan1 Metrica dalam dua kondisi. Semua telah terekspos dalam campaign sebelumnya.6 paired stress memakai match SkillCorner yang sama dengan transformasi tambahan, bukan6 match independen baru. Skor publik sebelumnya juga sudah diketahui; ini riset adaptif, bukan test buta baru.

## Holdout setelah selection freeze

| Cohort | Kontrol wave4 | Selected wave5 | Selisih mean spellF1 |
|---|---:|---:|---:|
| IDSSE2 | 0.644282 | 0.667175 | +0.022893 |
| SkillCorner6 clean | 0.676204 | 0.678304 | +0.002100 |
| SkillCorner6 stress | 0.664055 | 0.663030 | -0.001025 |
| Metrica3 dua kondisi | 0.657178 | 0.690016 | +0.032838 |

Pada10 scenarios biasa, selected menang8/10; dua kondisi Metrica3 bukan dua pertandingan independen. Bila dirata-ratakan per unique match, selected menang7/9. Kenaikan Metrica3 dibanding wave4 belum melampaui mean wave3 historis0.701784. Tidak ada klaim mengungguli semua campaign pada semua domain.

Pada6 paired stress, selected hanya menang2/6. Mean spellF1 sedikit turun; jangan memilih ulang decoder atau kandidat berdasarkan hasil ini. Seluruh64 role/scenario rows, termasuk duplicate-role provenance: [wave5_holdout.json](../results/wave5_holdout.json).

### Transformasi bola hilang

Sesudah freeze, seed20261113+identifier match tetap, terpisah per babak. Tambahan35% baris bola yang semula terlihat dihilangkan dalam contiguous spans10..60sampel. Label, pemain dan timeline diperiksa identik; seluruh X16 dan fitur per-player/ball dihitung ulang. Tidak sekadar mengosongkan kanal ball sambil membiarkan fitur jarak lama bocor.

| Match | Bola terlihat sebelum | Bola terlihat sesudah | Bagian visible yang dihapus |
|---|---:|---:|---:|
| skillcorner_2010085 | 66.24% | 43.05% | 35.001% |
| skillcorner_2011166 | 54.38% | 35.35% | 35.000% |
| skillcorner_2013725 | 64.63% | 42.01% | 34.999% |
| skillcorner_2015213 | 63.50% | 41.27% | 35.002% |
| skillcorner_2016236 | 61.01% | 39.65% | 34.999% |
| skillcorner_2017461 | 56.46% | 36.70% | 35.000% |

Ringkasan provenance tanpa posisi mask per frame: [wave5_stress_summary.json](../results/wave5_stress_summary.json). Prediksi GRU stress dibuat pada CPU dalam proses terpisah dari TCN, memakai checkpoint wave3 frozen. Input/feature/prior SHA diverifikasi lagi sebelum holdout/public.

## Short-spell diagnostics: tidak mengganti metric utama

Semua GT H/A spells tetap masuk scorer utama. Tabel memakai one-to-one same-team IoU matching resmi pada threshold0.75 untuk GT **<4detik**. Denominator kosong dilaporkan null, bukan dianggap recall sempurna. Recall berikut dipool dari counts dalam cohort untuk diagnosis, bukan objective provider-balanced atau confidence interval. Metrica3 memakai dua kondisi GT yang sama, sehingga254 bukan254 short spells independen.

| Cohort | GT short | Matched wave4 | Matched wave5 | Recall wave4 | Recall wave5 |
|---|---:|---:|---:|---:|---:|
| IDSSE2 | 360 | 55 | 64 | 15.28% | 17.78% |
| SkillCorner6 clean | 535 | 92 | 116 | 17.20% | 21.68% |
| SkillCorner6 stress | 535 | 85 | 100 | 15.89% | 18.69% |
| Metrica3 dua kondisi | 254 | 39 | 49 | 15.35% | 19.29% |

Short recall membaik di semua cohort itu, tetapi ada tradeoff pada stress:

| Stress pooled IoU0.75 | Wave4 | Wave5 |
|---|---:|---:|
| Predicted possessions | 1613 | 1713 |
| Matched possessions | 744 | 751 |
| Precision | 46.13% | 43.84% |
| Recall GT >=15detik | 71.98% | 68.64% |

Pipeline menghasilkan lebih banyak possessions, tetapi sebagian tambahan bukan match GT; short recall saja tidak menjamin spellF1 total membaik. Ini diagnosis pipeline lengkap, bukan efek terisolasi loss weighting, karena decoder juga berubah. Detail semua duration buckets/IoU dan timing error: [wave5_duration_diagnostics.json](../results/wave5_duration_diagnostics.json).

## Publik: tidak ada rekor spellF1 baru

Empat role fixed baru dievaluasi setelah freeze, holdout dan independent refit. `best_new` identik dengan `selected`, jadi hanya3 CSV berbeda. GT publik7.500frame/17 possessions;12 predicted possessions pada ketiga CSV. Tidak ada96-way public sweep atau pemilihan ulang dari macroF1 publik.

| CSV berbeda | SpellF1 | MacroF1 | Predicted possessions |
|---|---:|---:|---:|
| selected | 0.772414 | 0.902059 | 12 |
| previous_wave4 | 0.772414 | 0.895984 | 12 |
| pure_boundary | 0.772414 | 0.917189 | 12 |

`pure_boundary` memiliki macroF1 publik lebih tinggi daripada selected, tetapi validasi eksternalnya lebih rendah dan tidak lolos aturan adopsi. Kandidat tidak diganti sesudah mengetahui hasil tersebut. Skor public teratas yang teramati tetap sama dengan wave4; skor private/full-match belum diketahui. [Audit publik](../results/wave5_public.json).

## Verifikasi runtime dan reproduksi

- 10 regression tests lulus: variable-cost DP vs exhaustive optimum, dead interval reset, post-merge count calibration, boundary reset, unknown-label boundary supervision, full-spell short weights, visibility smoothing antar babak, learned-boundary provenance, short GT retention, dan unknown/half diagnostic splitting.
-  Actual CUDA fitting/inference/checkpoint/refit ketiga model lolos dengan4steps/seed sebagai smoke saja. Saved probabilities dan independent smoke refit identik; ini tidak dicampur dengan hasil recipe penuh.
-  Kontrol lama diulang pada23 development scenarios sebelum full training dan menghasilkan objective persis sama, bukan mendekati skor lama.
-  Full training21 ensemble fits/**63 seed fits**,94.500optimizer steps,CUDA,proses exit0. Waktu development/full training tercatat5037,110detik.
-  Sesudah selection freeze, komponen short-TCN terpilih benar-benar dilatih ulang dengan3seed/1500steps pada19match yang sama; GRU lama frozen diverifikasi hash, tidak diklaim dilatih ulang di wave5.
-  Independent rebuild227,094detik menghasilkan **199.319baris target identik byte-for-byte**. CSV kontrol wave4 juga identik dengan hash historis. Selection dan seluruh postfreeze input hashes tetap sama sampai audit publik.

CSV kandidat lokal: `wave5/out/candidates/selected.csv`.

SHA-256 CSV:

`cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`

SHA-256 frozen selection:

`6d69b0e4c702144a0ef2877c1522c2c6478f0ca5bb6b7089f315526bfc367b95`

Bukti agregat tambahan: [execution](../results/wave5_execution.json), [verification](../results/wave5_verification.json). Cara menjalankan ulang: [README.md](README.md). Source/PLAN/input dibekukan sebelum real fits; perubahan berikutnya harus menjadi campaign baru.

## Publikasi dan kesimpulan

Raw tracking, mask/label/probabilitas per frame, model weights dan submission CSV tetap ignored. Hanya kode dan metrik/provenance agregat disiapkan untuk dibagikan. Tidak ada commit/push otomatis. Seluruh file wave4 dan eksperimen anggota lain tidak diubah; indeks dokumen induk diperbarui.

SkillCorner MIT, IDSSE CC BY4.0. Kredit dan provenance Ryu/data diwarisi dari [wave4/reference/SOURCE.md](../wave4/reference/SOURCE.md). Wave5 tidak menambah training Metrica atau memakai anchor tree; Metrica tetap evaluasi proxy event, bukan bukti kesetaraan corrected GT target.

Short-loss+decoder baru layak dilanjutkan dari validasi eksternal, tetapi belum memecahkan long/short precision tradeoff saat bola hilang. Pengujian berikutnya perlu menjaga long possessions dan memakai fresh external validation bila tersedia; jangan memaksa kenaikan pada17 GT possessions publik yang sama. Tidak ada jaminan keunggulan private dari hasil ini.
