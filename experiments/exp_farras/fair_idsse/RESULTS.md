# Benchmark IDSSE bersama: short-loss PP-TCN tertinggi secara mean

Run selesai seluruh7folds,63neural seed fits,31500optimizer updates,7HGB fits. Tidak ada skor parsial atau checkpoint lama yang dicampur. Proses terpisah mereproduksi seluruh63decoded arrays identik. [Protokol sebelum run](PLAN.md), [reproduksi](README.md), [semua hasil](../results/fair_idsse_results.json).

## Hasil utama: decoder sama

Seluruh learned arms mendapat prepared r0 inputs dan provider GT yang sama,7leave-one-match-out folds, masing-masing training persis6match lain. Neural memakai seeds0/1/2,500updates/seed,batch16,window384. Semua probabilitas memakai decoder wave5 dead45 tanpa tuning per model/per outer fold. Tidak ada target/public GT.

| Keluarga adapted | Mean spellF1 | Mean macroF1 | Mean count error |
|---|---:|---:|---:|
| Short-loss PP-TCN, Farras-style | **0.595755** | **0.902935** | -23.98% |
| SetSeq, Jeremy-style | 0.578602 | 0.896519 | -32.51% |
| PP-TCN, Ryu-style | 0.571090 | 0.900344 | -33.82% |
| HGB symmetric, Farras wave2-style | 0.315719 | 0.771451 | +1.13% |

Count mendekati GT tidak menjamin possessions benar. HGB memiliki mean count error kecil tetapi spellF1 jauh di bawah neural models. Jadi hasil rendah HGB bukan hanya masalah tabel lama: pada benchmark baru pun keluarga model lama ini tertinggal. Pipeline temporal kita yang lebih baru adalah pembanding berbeda dari HGB lama.

**Ini bukan ranking exact submission tiap branch.** SetSeq adalah adapter wave3, bukan Jeremy exp7; PP-TCN dilatih ulang pada6IDSSE saja, bukan checkpoint Ryu27match; short-loss PP-TCN tidak memakai campuran frozen GRU wave5/6. Tidak boleh mengubah nama tabel menjadi ranking final anggota tim.

### Paired uncertainty

- Short-loss PP-TCN versus PP-TCN: gain0.024666, exploratory paired-bootstrap95%[0.018060,0.032109], menang7/7matches.
- Short-loss PP-TCN versus SetSeq: gain0.017153, interval[-0.003553,0.038950], menang5/7matches. Interval melintasi0, sehingga belum bukti kuat short-loss mengungguli SetSeq pada unseen matches.
- SetSeq versus PP-TCN: gain0.007513, interval[-0.018959,0.030560], menang5/7matches; belum terpisah jelas.

Bootstrap5000draws,seed20261114,unit unique match, bukan frame atau seed. Semua10pairs tersedia dalam JSON; interval exploratory dan tidak dikoreksi multiple comparisons. Tujuh match telah terekspos penelitian sebelumnya, bukan fresh blind test.

## Rule baseline terpisah

Touch-hold adapted menghasilkan spellF1 **0.200755**, macroF1 **0.600833**, mean count error+31.74%. Ini panel native hard-label rule, bukan shared probabilistic decoder. Rule memakai5fps tracking terkorupsi yang sama, metadata tim tanpa GT flip, speed posisi dalam canonical metric coordinates, tanpa oracle dead/X mask. Angka ini **bukan reproduksi** skor Rasya0.5205 pada tabel lama.

Tidak menyimpulkan seluruh code/submission Rasya hanya mampu0.2008. Perbedaan5fps adaptation, cues fixed, missingness, dan penghapusan GT-dependent orientation dapat mengubah hasil. Tidak ada ablation di sini yang mengisolasi kontribusi masing-masing perubahan.

## Argmax diagnostic, bukan pemilihan ulang

| Keluarga | SpellF1 argmax | MacroF1 argmax |
|---|---:|---:|
| HGB | 0.041246 | 0.717949 |
| SetSeq | 0.567694 | 0.898332 |
| PP-TCN | 0.560026 | 0.900524 |
| Short-loss PP-TCN | 0.571138 | 0.901641 |

Semua memakai argmax yang sama atas probabilitas3classes; hasil menunjukkan mengapa decoder temporal penting terutama bagi HGB. Tidak memakai hasil ini untuk memilih decoder khusus tiap model. Main table tetap decoder yang ditetapkan pada PLAN.

## Apa yang sekarang lebih setara

1. Semua learned fits mendapat pool6match yang sama dan mengeluarkan match uji seluruhnya. Tidak ada truncation first2matches seperti adapter HGB consolidation lama.
2. Tidak memuat shipped Ryu checkpoint yang pernah melihat tujuh match benchmark. Semua21neural ensemble fits dimulai dari nol.
3. Predictor hanya mendapat observasi; y/y25 dihapus. Neural API menolak training/test overlap serta label keys. Metadata home/away tidak dibalik berdasarkan GT heldout. Rule tidak menyalin GT dead/X mask.
4. GT provider asli sama, semua short spells masuk scorer resmi, timeline/half mapping sama untuk seluruh output.
5. Normalisasi model dan rate/dead_share decoder dihitung dari training fold saja. Decoder learned arms identik; rule-native dipisahkan jelas.
6. Neural updates/seeds/batch/window sama; budget model-specific HGB dan semua optimizer/loss/sampling tetap dilaporkan, bukan disamarkan sebagai causal architecture-only ablation.
7. Source/PLAN/data/features dibekukan sebelum full fits. Cache baru mengikat manifest, training membership, heldout,family,seeds,steps dan prediction SHA; mismatch gagal.

Benchmark tidak menjamin seluruh pilihan desain optimal untuk setiap arsitektur. Common decoder pun dapat lebih cocok pada sebagian keluarga. Budget500updates lebih rendah daripada1500step produksi; hasil bukan ceiling model. Tidak boleh membandingkan angka main table langsung dengan public spellF1 wave6 atau provider-balanced objective tiga domain wave5/6.

## Runtime dan verification

- 32 semantic tests lulus sebelum fits:6match exclusion, GT stripping, half-local rate, actual HGB pooling/fit, rule visibility/meter-per-second/reset/metadata, neural train/test isolation dan averaging joint probabilities.
- Smoke gabungan benar-benar selesai menghitung tetapi Windows keluar fatal -1073740791. SetSeq-only CUDA child juga keluar3221226505. Kedua run itu dicatat gagal, bukan diterima karena ada output JSON.
- Backend sebelum full freeze: SetSeq CPU; PP/short-TCN CUDA; HGB/rule CPU. Isolated smoke family yang diperbaiki berhasil exit0. Equal updates bukan equal hardware/FLOPs; tidak ada ranking latency/efisiensi.
- Full benchmark plus independent rebuild selesai exit0. Waktu script full15565.016detik; sebagian besar training SetSeq CPU (aggregate fit13791.594detik). Seluruh model/fold berjalan terpisah sehingga child exit0 diverifikasi sebelum parent membaca cache.
- 35 job membership records diperiksa: learned actual_fit_names tepat6nama lain, rule actual_fit_names kosong.63score rows mencakup35primary rows dan28argmax diagnostics.63full neural seed fits tidak mencakup reduced smoke fits, baik smoke yang gagal runtime maupun yang diterima.
- Independent process decode rebuild memuat ulang35prediction caches dan mereproduksi63hash label arrays identik. Ini bukan independent model refit dan bukan199319-row target rebuild, karena target tidak dievaluasi.

Manifest SHA: `9e47a83be188a497aa2c0b6f4e02f0f231235d94c65dcd61b2f816ecd378dff9`.

[Protocol hashes](../results/fair_idsse_protocol.json), [runtime/isolation checks](../results/fair_idsse_checks.json), [decode rebuild](../results/fair_idsse_verification.json).

## Koreksi audit fitur sebelumnya

Klaim sebelumnya bahwa model kita otomatis mewarisi nearest-opponent bug salinan referensi Ryu tidak tepat. File referensi memang memakai mask relatif view, tetapi builder aktual `wave4/features.py:_player_set` memakai per-player team inequality dan menolak >48pemain hadir. Benchmark memakai feature packs builder aktual, bukan fungsi referensi itu. Tidak ada perbaikan algoritme wave4/5/6 dalam run ini; yang dikoreksi adalah penjelasan audit, bukan hasil frozen lama.

## Publikasi

Kode baru hanya `fair_idsse/`; raw tracking, probabilities/labels per frame, weights serta caches tetap ignored. Hanya4aggregate JSON diekspor ke `results/`. Model baru tidak dipush dan tidak dijadikan kandidat submission otomatis. Private/full-match score belum diketahui. Source teman dan campaign lama tidak dimodifikasi; indeks Farras mencatat benchmark baru dan batas perbandingannya.
