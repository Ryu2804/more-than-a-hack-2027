# Wave4: PP-TCN lintas provider dengan referensi Ryu

Kandidat dibekukan melalui validasi eksternal: **`pp_gru25/no_shift`**. Skor publik meningkat dari wave3, tetapi PP-TCN murni menyamai spellF1 publiknya. Bukti peningkatan terbesar berasal dari data lintas provider dan representasi per-player, bukan bukti bahwa campuran GRU wajib dipakai. Skor private/full-match belum diketahui.

## Publik: empat kandidat fixed setelah freeze

Scorer resmi, `bootstrap_n=0`, window GT publik7.500 frame dengan17 GT possessions. Tidak ada pencarian64 konfigurasi pada GT publik.

| Kandidat | SpellF1 | MacroF1 | Predicted possessions |
|---|---:|---:|---:|
| selected | 0.772414 | 0.895984 | 12 |
| new_pp_tcn | 0.772414 | 0.899359 | 12 |
| previous_wave3 | 0.575610 | 0.712844 | 24 |
| shipped_ryu | 0.760000 | 0.919932 | 13 |

`selected` berarti75% probabilitas PP-TCN baru +25% GRU hybrid wave3, bukan75% bobot checkpoint Ryu yang sudah melihat holdout. `new_pp_tcn` adalah model baru murni dengan decoder `ryu_rate`; `previous_wave3` adalah kandidat frozen wave3, bukan wave3 yang dituning ulang. Kedua kandidat baru memiliki spellF1 publik sama meskipun macroF1 dan label per frame berbeda. Pilihan tidak diganti setelah melihat macroF1 publik.

Kandidat terpilih memprediksi12 dibanding17 GT possessions: under-count29,41%. MacroF1 lebih rendah daripada adapter shipped Ryu. Selisih spellF1 pada satu window kecil bukan bukti keunggulan statistik atau private.

## Audit branch Ryu

Source dipin pada [`5eaf3e595e91fa0502aac48037d0cae8c356ec6f`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/5eaf3e595e91fa0502aac48037d0cae8c356ec6f). `017_multi_match/NOTES.md` melaporkan sekitar0.747 publik untuk player-set TCN full-mix; technical document lama0.422 bukan hasil terbaru. Catatan grouped6-fold30-match melaporkan sekitar0.661 tanpa player-set dan0.679 dengan player-set. Angka filtered yang membuang short possessions tidak dijadikan metrik utama campaign ini.

Arsitektur yang diambil: pooled player features dan dua team views, TCN residual6 blok, kernel5, channel64, dilations `(1,2,4,8,16,1)`, GroupNorm, serta dua head dead/team. Player-set menambah MLP per pemain, attention pooling dan max pooling. Decoder menilai pergantian H/A dengan gate jarak bola, rate calibration dari training, merge live slivers, dan dead overlay yang terpisah.

Checkpoint shipped `weights_nometrica` ternyata dilatih pada **seluruh27 pertandingan:7 IDSSE +20 SkillCorner**. Karena itu checkpoint tersebut dilarang menjadi prediksi held-out untuk IDSSE/SkillCorner, dan tidak dipakai memilih64 konfigurasi. Referensi shipped hanya dievaluasi pada Metrica3 dan target. Metadata report checkpoint memberi provider-balanced rate3.5828524 possessions/menit dan dead share0.4574683; bukan statistik training model baru atau asumsi bahwa angka0.747 berlaku persis pada checkpoint ini.

Skor shipped pada tabel adalah **adapter**, bukan reproduksi byte-exact CSV asli Ryu. Perbedaannya: input target berasal dari cache lokal, slot tim yang tidak diketahui21/22/43 dimask, prediksi/decoder memakai grid5fps lalu label diulang25fps, rate memilih grid lambda terdekat alih-alih log interpolation, dan tidak memakai forced dead interval khusus kickoff target. Feature code25fps dan tiga state dict Ryu tetap asli. Lima snapshot source tidak diedit; provenance di [reference/SOURCE.md](reference/SOURCE.md).

## Data, pemisahan, dan protokol

- 20 pertandingan SkillCorner diunduh dari upstream publik dengan validasi ukuran dan SHA-256 LFS. Raw tracking1.823.448.331 byte; timeline25fps3.112.002 frame,591.139 sampel5fps di interval babak.
-  Training baru hanya19 pertandingan:5 IDSSE development dan14 SkillCorner development. IDSSE memiliki tiga corruption rounds; provider, match, round, lalu crop disampling seimbang saat training.
- 6 grouped folds mengecualikan seluruh round pertandingan evaluasi. Jumlah pertandingan training per fold15/15/16/16/16/17. Metrica1/2 clean/degraded merupakan zero-shot stress untuk TCN dan GRU, bukan sumber training kedua model itu.
-  Objective: mean spellF1 tiap match, clean/degraded dirata-ratakan dalam match, lalu mean tiap provider dengan bobot sama untuk IDSSE/SkillCorner/Metrica. MacroF1 tie-breaker. Count error dicatat, tidak dijadikan objective tersembunyi.
- 23 development scenarios berasal dari21 pertandingan:5 IDSSE,14 SkillCorner,2 Metrica dengan dua kondisi masing-masing.64 konfigurasi menghasilkan1.472 evaluasi, bukan1.472 model independen.
-  Holdout campaign:2 IDSSE,6 SkillCorner, danMetrica3 clean/degraded, jadi9 pertandingan unik dalam10 scenarios. IDSSE/Metrica3 pernah terekspos di campaign sebelumnya; SkillCorner6 baru untuk Farras tetapi sudah dipakai riset Ryu. Tidak disebut globally blind.
-  Seluruh training baru dan inferensi lokal. GT target tetap tidak dipakai training/seleksi. Registry mengunci source, input, checkpoint prior, split, recipe dan device sebelum fit; seleksi dihash sebelum holdout dan public.

Aturan, nama match, decoder, dan recipe sebelum hasil tersedia: [PLAN.md](PLAN.md). Mean development wave3 sebelumnya tidak dibandingkan langsung dengan mean wave4 karena dataset dan pembobotan berubah.

## Hasil development dan ablasi

Tiga arsitektur baru: pooled TCN, player-set TCN, dan player-set+7 control channels. Ensemble3 seed,1500 optimizer steps/seed, batch16, crop384, AdamW learning rate0.002/weight decay0.01, OneCycle, dropout0.15. Setiap fold/full model dilatih terpisah. Ball masking juga menutup kanal PP terkait bola dan seluruh7 control channels agar model tidak mendapat jalur bocor.

Delapan family probabilitas dikombinasikan dengan delapan decoder. Tabel menampilkan konfigurasi terbaik setiap family dan satu kontrol matched `ryu_rate` untuk family terpilih. Seluruh64 konfigurasi serta metrik per match tersimpan pada [wave4_development.json](../results/wave4_development.json).

| Family | Decoder | Mean spellF1 | Mean macroF1 | IDSSE spellF1 | SkillCorner spellF1 | Metrica spellF1 |
|---|---|---:|---:|---:|---:|---:|
| plain_tcn | no_shift | 0.609125 | 0.875965 | 0.516401 | 0.683777 | 0.627198 |
| pp_tcn | no_shift | 0.649605 | 0.881666 | 0.570895 | 0.707571 | 0.670350 |
| control_tcn | no_shift | 0.650108 | 0.881234 | 0.570460 | 0.707524 | 0.672340 |
| pp_gru25 | no_shift | 0.656490 | 0.884522 | 0.583755 | 0.709044 | 0.676673 |
| pp_gru50 | no_shift | 0.650717 | 0.883483 | 0.586995 | 0.676695 | 0.688462 |
| pp_gru75 | no_shift | 0.633341 | 0.874851 | 0.577774 | 0.626968 | 0.695281 |
| pp_tree25 | no_shift | 0.646579 | 0.881659 | 0.567203 | 0.705694 | 0.666839 |
| prior_combined | no_shift | 0.614330 | 0.865600 | 0.565785 | 0.592999 | 0.684204 |
| pp_gru25 | ryu_rate | 0.653885 | 0.883764 | 0.585623 | 0.704171 | 0.671859 |

Player-set mengalahkan pooled TCN pada21/21 pertandingan development, dengan decoder matched `no_shift`. Penambahan7 control channels pada TCN hanya memberi kenaikan kecil; tidak ada dasar menyebutnya peningkatan kuat. Campuran25% GRU memberi kenaikan development kecil terhadap PP-TCN, di bawah adoption margin0.01 yang disebut riset Ryu. Campaign ini memakai aturan ranking fixed pada PLAN, bukan mengklaim campuran sudah lolos aturan adopsi Ryu atau lolos uji signifikansi.

`no_shift` berarti tidak memaksa dead prior menuju0.407; bukan menghapus dead head/dead overlay. Parameter lain: gate0.35m, kappa0.05, merge1s, dead threshold0.5, minimum dead run2s, tau3s, rate scale1. Training-only provider-balanced rate final3.6428176 possessions/menit, dead share0.4608204. Semua family terbaik memilih `no_shift`, tetapi GT penuh target belum tersedia untuk membuktikan generalisasi keputusan prior itu.

Koreksi fitur: kecepatan dihitung pada5fps nyata; rolling2s/median1s reset per babak; jarak lawan untuk tiap pemain menggunakan tim lawan pemain tersebut, bukan shortcut tim lawan view yang dapat memasukkan dirinya sendiri. Missing observations tidak diubah menjadi bukti gate jarak yang valid. Kalibrasi menghitung seluruh grid lambda, tanpa asumsi bahwa jumlah possessions selalu monoton setelah decoding.

## Holdout setelah seleksi

Mean spellF1 per provider. Kolom Metrica adalah dua kondisi pertandingan yang sama, bukan dua pertandingan independen.

| Kandidat | IDSSE2 | SkillCorner6 | Metrica3 clean/degraded |
|---|---:|---:|---:|
| selected | 0.644282 | 0.676204 | 0.657178 |
| new_pp_tcn | 0.632643 | 0.672262 | 0.639989 |
| previous_wave3 | 0.597654 | 0.585117 | 0.701784 |
| shipped_ryu | tidak sah sebagai holdout | tidak sah sebagai holdout | 0.635642 |

Pada8 pertandingan IDSSE/SkillCorner holdout, kandidat terpilih menang7/8 terhadap wave3 dan7/8 terhadap PP-TCN default. Kenaikan mean per-match terhadap wave3 adalah0.079972, tetapi perbandingan ini mengubah data training sekaligus arsitektur/decoder; bukan efek terisolasi campuran GRU. Worst absolute count error SkillCorner turun dari48,93% wave3 menjadi6,33%; IDSSE masih29,73%. Development worst count error terpilih tetap44,62%, sehingga tidak memenuhi count-error30% pada semua kasus.

Metrica3 **regresi terhadap wave3**, pada kondisi clean maupun degraded. Rata-rata provider baru tidak menghapus kegagalan domain tersebut. Rincian10 scenarios:

| Match/kondisi | Selected | PP-TCN default | Wave3 frozen |
|---|---:|---:|---:|
| J03WQQ/r0 | 0.654913 | 0.636805 | 0.657420 |
| J03WR9/r0 | 0.633651 | 0.628481 | 0.537888 |
| metrica3/clean | 0.671637 | 0.653722 | 0.740230 |
| metrica3/r0 | 0.642718 | 0.626256 | 0.663339 |
| skillcorner_2010085/clean | 0.667532 | 0.669623 | 0.568276 |
| skillcorner_2011166/clean | 0.700952 | 0.698252 | 0.562940 |
| skillcorner_2013725/clean | 0.683980 | 0.676819 | 0.595556 |
| skillcorner_2015213/clean | 0.685333 | 0.675433 | 0.594564 |
| skillcorner_2016236/clean | 0.660929 | 0.658699 | 0.595930 |
| skillcorner_2017461/clean | 0.658494 | 0.654745 | 0.593438 |

Shipped Ryu pada Metrica3: clean0.651803, degraded0.619481. Hanya dua rows itu valid sebagai external reference karena seluruh IDSSE/SkillCorner checkpoint telah menjadi training source. Rincian32 rows total dan audit adapter: [wave4_holdout.json](../results/wave4_holdout.json).

## Runtime dan reproduksi yang benar-benar dijalankan

- 11 regression tests lulus: exhaustive DP, minimum dead run/half reset, live sliver, lambda nonmonotonic, known/visible ball gate, partition validation, conditional fusion, PP opponent/permutation, unit/half reset, mask control/PP, dan penolakan shipped reference pada provider in-sample.
-  Smoke actual fitting/predict/checkpoint/refit ketiga arsitektur lolos. CUDA Conv1d exit0; masalah shutdown GRU CUDA lokal dari wave3 tidak dibawa ke training TCN. Prior GRU disiapkan pada CPU dalam proses terpisah.
-  Campaign development/full training:21 ensemble fits,**63 seed fits** dengan recipe penuh,94.500 optimizer steps; device CUDA, proses exit0. Waktu tercatat4082,453 detik.
-  Sesudah freeze, PP-TCN terpilih dilatih ulang dari seed pada19 match yang sama,221,735 detik. Komponen GRU frozen tidak dilatih ulang di wave4; checkpoint/input-nya dihash dan bukti refit wave3 tetap berlaku. Campuran dan decoder menghasilkan **199.319 label target identik byte-for-byte** dengan kandidat awal.
-  CSV wave3 juga direproduksi dengan hash historis yang sama sebelum menjadi pembanding. Hash selection tetap sama setelah holdout/public.
-  Pinned feature code asli Ryu mengeluarkan `RuntimeWarning: Mean of empty slice` dan `Degrees of freedom <= 0 for slice.` pada missing observations; warning tidak disembunyikan. Adapter tetap menghasilkan probabilitas finite dan empat CSV lolos scorer. Tidak ada klaim runtime tanpa warning.

CSV kandidat terpilih lokal: `wave4/out/candidates/selected.csv`.

SHA-256 CSV:

`ddd75416e7929183954d27191b11ddab1ac049fc0e703f802db2209884635e9a`

SHA-256 frozen selection:

`b5269672e86cce891c7ce28054ce17a6acfa4c4979e11236757212d908acdd05`

Agregat yang aman dibagikan: [execution](../results/wave4_execution.json), [verification](../results/wave4_verification.json), [public](../results/wave4_public.json), development dan holdout di atas. Raw data, probabilitas/label per frame, model weights, cache dan CSV submission tetap di folder ignored. Tidak ada commit/push otomatis.

## Lisensi dan batas kesimpulan

[SkillCorner opendata](https://github.com/SkillCorner/opendata) berlisensi MIT. IDSSE: [Bassek et al., Scientific Data12,195(2025)](https://doi.org/10.1038/s41597-025-04505-y), [data CC BY4.0](https://doi.org/10.6084/m9.figshare.28196177). Provenance dan kredit source Ryu tidak dihapus.

TCN baru dan GRU hybrid pada kandidat terpilih tidak dilatih pada Metrica. Family `pp_tree25` dan `prior_combined` memakai anchor tree lama yang dilatih pada Metrica; evaluasi Metrica1/2 untuk anchor mengecualikan game yang dinilai, tetapi ketidakjelasan lisensi training anchor tetap ada. Metrica dipakai sebagai evaluasi zero-shot, bukan otomatis dianggap berlisensi eksplisit.

Data eksternal menggunakan proxy event/phases, tidak identik definisi corrected GT kompetisi. Pencarian64 konfigurasi memakai development yang sama, augmentasi tidak menjamin noise setara target, dan beberapa holdout pernah terekspos pada campaign/riset lain. Model memakai konteks temporal masa depan dan berjalan offline. Peningkatan publik dan SkillCorner holdout nyata pada evaluasi ini; keunggulan full/private belum terverifikasi. Prioritas uji berikutnya adalah fresh external validation dan short-possession/occlusion robustness tanpa membuang short GT dari metrik.
