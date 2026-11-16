# Hasil wave3: kombinasi Jeremy dan Farras

## Keputusan

66 konfigurasi selesai:18 neural fits,18 IDSSE tree fits,594 development evaluations,16 holdout candidate/scenario evaluations, dan4 public sanity checks. Total tiga batch Farras:138 konfigurasi dan882 development evaluations. Bukan138 model independen; beberapa decoder/mixture memakai probabilitas model yang sama.

Kandidat dibekukan sebelum holdout/public: **hybrid_tree25 + centered2**. Probabilitas state75% model identity-free set+BiGRU dengan tambahan7 fitur kontrol Farras,25% tree kontrol yang dilatih pada Metrica. Boundary probabilities dicampur dengan bobot yang sama; anchor tree memakai boundary netral0.5. Decoder memakai lambda_team2,lambda_dead2,gamma0.5 dan temperature1.

Mean development spellF1 **0.615394**, macroF1 **0.873323**. Angka ini berasal dari ranking development atas66 konfigurasi, bukan estimasi test tanpa selection bias. Bukan perbandingan langsung dengan proxy Jeremy0.656 karena dataset split, model, token sampling, recipe, decoder, dan objective berbeda.

## Publik: naik dari Farras lama, belum melampaui Jeremy asli

| Pipeline | Public spellF1 | Public macroF1 | Predicted possessions |
|---|---:|---:|---:|
| Jeremy asli, laporan branch | 0.581818 | 0.725033 | 16 |
| Kombinasi frozen wave3 | 0.575610 | 0.712844 | 24 |
| Farras kontrol lama | 0.560000 | 0.650520 | 13 |
| Adaptasi SetSeq lokal + cost0.5 | 0.526316 | 0.739575 | 21 |
| Farras wave2 frozen | 0.460606 | 0.711830 | 16 |

Window memiliki17 GT possessions. Kandidat baru lebih tinggi0.015610 dari diagnostic Farras0.56, tetapi lebih rendah0.006208 dari laporan Jeremy0.581818. Tidak ada bukti signifikansi kedua selisih; interval Jeremy lebar dan window kecil. MacroF1 kandidat baru juga lebih rendah dari adaptasi SetSeq, walau spellF1 lebih tinggi. Jumlah24 possessions menunjukkan over-segmentation pada window ini.

Hanya empat kandidat prespecified dinilai pada public setelah freeze. Tidak dilakukan66-way public sweep, perubahan parameter, atau penggantian pemenang sesudah melihat skor. Hasil publik tidak dianggap skor private/full-match dan tidak menjadi alasan mengadopsi kandidat ini menggantikan submission Jeremy.

## Holdout: perbandingan dengan input yang sama

Dua IDSSE holdout tidak dilatih atau dipakai memilih konfigurasi campaign ini. Namun keduanya pernah dipakai riset Jeremy. Metrica3 sudah pernah dievaluasi pada wave2. Semua limit ini diketahui sebelum training.

| Skenario spellF1 | Kombinasi | Adaptasi SetSeq | Kontrol lama | Symmetric lama |
|---|---:|---:|---:|---:|
| IDSSE J03WQQ degraded | 0.657420 | 0.647263 | 0.249697 | 0.283692 |
| IDSSE J03WR9 degraded | 0.537888 | 0.501046 | 0.254669 | 0.294098 |
| Metrica3 asli | 0.740230 | 0.723864 | 0.589509 | 0.599664 |
| Metrica3 degraded | 0.663339 | 0.639759 | 0.359310 | 0.389950 |

Kombinasi mengungguli ketiga reference pada empat skenario. Mean dua IDSSE:0.597654 versus0.574155 untuk adaptasi SetSeq. Mean tiga pertandingan, dengan clean/degraded Metrica3 dirata-ratakan dulu:0.632364 versus0.610040. Hanya tiga pertandingan dan tidak ada paired-CI yang layak untuk klaim kemenangan populasi.

Metrica3 degraded memakai degrader Jeremy dengan player gaps, permutation, duplicate slots dan ball gaps, bukan oklusi ball-only wave2. Karena kondisi berbeda, jangan membandingkan langsung0.663339 dengan skor occluded wave2 lama0.433982.

## Apa yang dipelajari dari ablasi

Semua delta berikut development, bukan independent holdout ablation. Tidak ada perubahan grid setelah melihat hasil.

- SetSeq+cost0.5:0.607472. Hybrid dengan7 fitur kontrol pada decoder yang sama:0.609823. Tambahan kontrol memberi delta kecil0.002351, bukan lonjakan yang sudah terbukti stabil.
- Pada centered2 yang sama, hybrid0.613449 dan hybrid_tree25 0.615394: tambahan25% tree memberi0.001946. Kontribusi mixture kecil pada development, walau kandidat lengkap lebih baik pada holdout.
- Pada hybrid_tree25 yang sama, constant-cost2 memberi0.603684 dan centered2 memberi0.615394: delta0.011710. Ini mendukung mencoba centered boundary dalam campaign baru, tidak membatalkan laporan Jeremy bahwa follow-up aslinya inconclusive.
- Best feature-only GRU0.398470, jauh di bawah best SetSeq0.610136. Keduanya dilatih pada fold yang sama, tetapi set model juga memakai mirroring; bukan isolasi arsitektur murni tanpa faktor lain.
- Tree anchor Metrica-only best0.306517 pada objective gabungan, meskipun masih0.56 public. Perbedaan domain training berarti ini bukan perbandingan kapasitas arsitektur yang sepenuhnya setara.
- Memperbesar tree weight ke75% menurunkan best score menjadi0.480214. Visibility gate0.534255 juga tidak mengalahkan bobot tetap25%. Hipotesis bahwa tree harus dominan ketika bola terlihat tidak didukung di protokol ini.
- Kandidat terpilih masih under-segment pada semua5 IDSSE development: worst count error -41.68%, gagal batas kualitas30% yang dicatat sebagai warning. Pada dua IDSSE holdout error -20.20% dan -25.95%, tetapi jumlah possession publik berlebih41.18%. Satu count prior global belum aman lintas provider.

## Ranking lengkap66 konfigurasi

Skor adalah rata-rata7 match:5 IDSSE leave-one-match-out dan2 Metrica zero-shot; clean/degraded dirata-ratakan di dalam tiap Metrica match terlebih dahulu. MacroF1 tie-breaker, count error tidak mengganti metric utama. Seluruh9 fold/scenario rows per konfigurasi tersedia di `../results/wave3_development.json`.

| # | Family | Decoder | Mean spellF1 | Mean macroF1 | Max abs count error |
|---:|---|---|---:|---:|---:|
| 1 | hybrid_tree25 | centered2 | 0.615394 | 0.873323 | 0.4168 |
| 2 | hybrid | centered2 | 0.613449 | 0.873529 | 0.3973 |
| 3 | hybrid_tree25 | team05_dead2 | 0.612492 | 0.872935 | 0.3875 |
| 4 | hybrid | team05_dead2 | 0.610887 | 0.872795 | 0.3836 |
| 5 | hybrid_tree25 | cost05 | 0.610821 | 0.872949 | 0.3855 |
| 6 | setseq | centered2 | 0.610136 | 0.870886 | 0.3855 |
| 7 | hybrid | cost05 | 0.609823 | 0.872685 | 0.3796 |
| 8 | setseq | team05_dead2 | 0.608521 | 0.870031 | 0.3699 |
| 9 | setseq | cost05 | 0.607472 | 0.870203 | 0.3620 |
| 10 | hybrid | cost2 | 0.606614 | 0.871993 | 0.4677 |
| 11 | hybrid_tree50 | team05_dead2 | 0.605708 | 0.869219 | 0.3777 |
| 12 | hybrid | team2_dead05 | 0.605204 | 0.871776 | 0.4638 |
| 13 | hybrid_tree50 | cost05 | 0.604500 | 0.868877 | 0.3659 |
| 14 | hybrid_tree25 | cost2 | 0.603684 | 0.872127 | 0.4736 |
| 15 | hybrid_tree25 | team2_dead05 | 0.602476 | 0.871954 | 0.4716 |
| 16 | hybrid_tree50 | centered2 | 0.601856 | 0.868013 | 0.4677 |
| 17 | setseq | cost2 | 0.600078 | 0.869230 | 0.4579 |
| 18 | setseq | team2_dead05 | 0.598662 | 0.869317 | 0.4481 |
| 19 | hybrid_tree50 | cost2 | 0.593213 | 0.866667 | 0.4912 |
| 20 | hybrid_tree50 | team2_dead05 | 0.590878 | 0.866401 | 0.4736 |
| 21 | hybrid | cost6 | 0.568256 | 0.868403 | 0.5205 |
| 22 | setseq | cost6 | 0.555624 | 0.866024 | 0.5264 |
| 23 | hybrid_tree25 | cost6 | 0.549742 | 0.866851 | 0.5499 |
| 24 | hybrid_visibility | centered2 | 0.534255 | 0.813619 | 0.3679 |
| 25 | hybrid_visibility | cost2 | 0.529367 | 0.812371 | 0.3992 |
| 26 | hybrid_tree50 | cost6 | 0.524832 | 0.858872 | 0.5969 |
| 27 | hybrid_visibility | team2_dead05 | 0.509922 | 0.810043 | 0.3131 |
| 28 | hybrid_tree75 | centered2 | 0.480214 | 0.737072 | 0.4501 |
| 29 | hybrid_tree75 | cost2 | 0.473812 | 0.735027 | 0.4618 |
| 30 | hybrid_visibility | cost6 | 0.469769 | 0.808332 | 0.5616 |
| 31 | hybrid_tree75 | team2_dead05 | 0.454333 | 0.731328 | 0.3562 |
| 32 | hybrid_visibility | team05_dead2 | 0.430402 | 0.807795 | 0.7050 |
| 33 | hybrid_visibility | cost05 | 0.422888 | 0.806285 | 0.7391 |
| 34 | hybrid_tree75 | cost6 | 0.413357 | 0.737900 | 0.6204 |
| 35 | feature_gru | cost05 | 0.398470 | 0.804888 | 0.5362 |
| 36 | feature_gru | team05_dead2 | 0.397944 | 0.804727 | 0.5421 |
| 37 | feature_gru | centered2 | 0.395107 | 0.806296 | 0.5695 |
| 38 | feature_gru | team2_dead05 | 0.392972 | 0.805419 | 0.6125 |
| 39 | feature_gru | cost2 | 0.391968 | 0.805451 | 0.6184 |
| 40 | hybrid_tree75 | team05_dead2 | 0.380696 | 0.731778 | 0.5652 |
| 41 | hybrid_tree75 | cost05 | 0.371290 | 0.727037 | 0.5963 |
| 42 | feature_gru | cost6 | 0.371048 | 0.804392 | 0.6556 |
| 43 | all_features | cost6 | 0.358719 | 0.769908 | 0.5540 |
| 44 | all_features | cost2 | 0.341599 | 0.751008 | 0.5458 |
| 45 | control_symmetric | cost2 | 0.338520 | 0.663530 | 0.4286 |
| 46 | all_features | centered2 | 0.334167 | 0.746242 | 0.7932 |
| 47 | control_symmetric | centered2 | 0.334117 | 0.653292 | 0.4881 |
| 48 | control | cost2 | 0.327532 | 0.644256 | 0.3699 |
| 49 | control | centered2 | 0.323729 | 0.638883 | 0.6475 |
| 50 | all_features | team2_dead05 | 0.311170 | 0.745424 | 1.2136 |
| 51 | anchor | cost6 | 0.306517 | 0.625730 | 0.5300 |
| 52 | control_symmetric | cost6 | 0.306346 | 0.676986 | 0.6771 |
| 53 | control | cost6 | 0.299590 | 0.660926 | 0.6673 |
| 54 | control_symmetric | team2_dead05 | 0.277299 | 0.630987 | 0.5932 |
| 55 | control | team2_dead05 | 0.276458 | 0.610189 | 0.7112 |
| 56 | anchor | centered2 | 0.273167 | 0.614331 | 0.4317 |
| 57 | anchor | cost2 | 0.273167 | 0.614331 | 0.4317 |
| 58 | anchor | team2_dead05 | 0.242419 | 0.609601 | 0.5528 |
| 59 | control_symmetric | team05_dead2 | 0.217938 | 0.643836 | 1.5864 |
| 60 | all_features | team05_dead2 | 0.208926 | 0.727328 | 2.5729 |
| 61 | control | team05_dead2 | 0.206338 | 0.616641 | 2.0407 |
| 62 | all_features | cost05 | 0.196705 | 0.722837 | 3.1797 |
| 63 | control_symmetric | cost05 | 0.190868 | 0.617068 | 1.8508 |
| 64 | control | cost05 | 0.189084 | 0.594735 | 2.2237 |
| 65 | anchor | team05_dead2 | 0.148035 | 0.594204 | 2.1429 |
| 66 | anchor | cost05 | 0.139193 | 0.590849 | 2.2267 |

## Reproduksi dan pemeriksaan

-34 prepared packs:21 IDSSE (7match x3round),12 Metrica (3match x4condition),1 target. Fitur, blocks, y/y25 alignment, dan unknown-team masks lulus.
-14 raw IDSSE metadata/tracking files, total2.622.686.298 byte, diverifikasi terhadap ukuran dan LFS SHA-256/Git object inventory PySport. Raw data tetap lokal.
- Target team mapping tidak ditebak dari urutan24/22 slot. Minimum20 votes dan99% purity per sampled-present slot;43 present-known slots dan3 empty sampled slots. Sebanyak37.651 finite team distances cocok persis;37 unavailable distances memiliki availability mask yang sama, tidak diimputasi sebagai observasi valid. Audit hanya sampled graded frames5fps, bukan klaim semua25fps team identities terverifikasi.
-6 regression tests lulus: exhaustive optimum enam decoder; reset/zero-probability/empty blocks; visibility fusion; permutation invariance dan mirror involution; control feature preservation dan per-half velocity; missing team-distance equivalence.
- Tiga tree families menjalani synthetic fit/predict, constant H/A/D, output normalization, label-free test, input immutability dan boundary reset.
- Ketiga neural families menjalani synthetic train/predict, independent deterministic refit, checkpoint roundtrip. cuDNN GRU Windows menghasilkan nilai deterministik tetapi process teardown crash -1073740791. CPU dan native CUDA selesai bersih; campaign memilih CPU single-thread tanpa menekan error atau mengurangi model/epochs.
-36 development model fits selesai dengan proses exit0. Empat target CSV lolos validator resmi. Kandidat dipilih tanpa target labels.
- Kandidat terpilih dilatih ulang independen pada semua5 development IDSSE dan anchor Metrica dengan seed yang sama. Semua199.319 target rows dan SHA-256 CSV identik.
- Hash seleksi identik antara holdout dan public report. Kode/input/checkpoints diverifikasi sebelum kedua fase.

CSV kandidat: `out/candidates/selected.csv`. SHA-256 `337734b6bf9b70bda6c9a16b549e39e3c90bdbd53fc6073c140ab07cd06b6096`.

Bukti agregat aman dibagikan: `../results/wave3_{development,holdout,public,execution,verification}.json`. Checkpoint, raw data, probabilitas dan CSV per-frame diabaikan Git. Tidak ada upload submission atau push otomatis untuk wave3.

## Eksperimen berikut yang masih layak, belum dijalankan

1. Reproduksi checkpoint/recipe Jeremy asli di backend PyTorch yang stabil sebelum menyebut kombinasi ini mengalahkan Jeremy. Adapter lokal lebih kecil dan bukan substitusi hasil aslinya.
2. Seed ensemble terpreregister pada data yang sama untuk mengukur stabilitas gain mixture0.002 dan centered decoder, bukan memilih seed dari window publik.
3. Tambah dataset holdout dengan possession/dead-ball berdefinisi dekat challenge. Domain cue off-pitch dan count rate berbeda; perlu validasi lintas provider sebelum count-aware decoder agresif.
4. Raw target bundle beserta metadata agar input pipeline diregenerasi tanpa cache pihak lain. Mapping nearest-distance yang terverifikasi bukan pengganti provenance raw metadata.

Semua66 konfigurasi dalam [PLAN.md](PLAN.md) sudah dijalankan. Daftar berikut ini keputusan riset setelah hasil, bukan bagian campaign yang ditinggalkan belum selesai. Tidak ada training/tuning lanjutan berdasarkan public diagnostic.

Audit sumber Jeremy: [JEREMY_REVIEW.md](JEREMY_REVIEW.md). Perintah reproduksi: [README.md](README.md).
