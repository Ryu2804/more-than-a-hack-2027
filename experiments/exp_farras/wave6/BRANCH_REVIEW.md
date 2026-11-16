# Audit seluruh branch, fetched 2026-11-14

`git fetch --all --prune` berhasil. Enam branch remote ditemukan; origin/HEAD bukan branch tambahan. Branch lokal main dan farras cocok dengan refs remote masing-masing saat inventaris. Tidak checkout, merge, commit atau push. Audit membaca source/report dan artefak agregat, bukan menjalankan ulang seluruh pipeline tim.

| Remote branch | Commit | Isi yang diperiksa |
|---|---|---|
| main | ef22d025b0c02bcdbcec53df34973ece0f53a701 | Baseline dan spesifikasi scorer |
| farras | 87f3236d4ed875f4cc868d9f750a0776a7779333 | Wave1/2 yang dipush; dibandingkan wave3/4/5 lokal |
| jeremy | ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1 | REPORT/decision/source exp_1 sampai exp_7 |
| rasya | a13249724b994728b5686d01bc48e6647cd1e40e | CONTEXT, TECHNICAL, parser, cue dan pipeline012 |
| ryu | 5eaf3e595e91fa0502aac48037d0cae8c356ec6f | NOTES015/016/017, features, training, shipped decoder |
| consolidation | 264efa67e1a2e74cce22fea81ed784acf159b258 | Registry, shared scorer/runner, adapter empat anggota dan summary |

## Perbaikan paling penting

### 1. Registry consolidation salah memberi nama beberapa angka

`experiments/CANDIDATES.md` menaruh0.408/0.410/0.488 sebagai public spellF1 Jeremy exp2/3/5. Source branch Jeremy menunjukkan angka itu cocok dengan **target dead share**, bukan skor terhadap GT publik:

- exp2 `results/target_candidate.json`: D share sekitar0.407543, `public_labels_used:false`.
- exp3 `results/target_candidate.json`: dead_share0.41040684389104665, `public_labels_used:false`.
- exp5 `results/target_candidates.json`: dead_share0.48837004349824054, `public_labels_used:false`.

Public spellF1 yang benar-benar tercatat pada laporan Jeremy adalah exp1 **0.581818**, macroF1 **0.725033**. Registry perlu membedakan prediction statistics, proxy evaluation dan public evaluation. Tidak menyalin angka registry yang keliru ke perbandingan wave6.

### 2. Common benchmark bukan ranking pipeline terbaru yang setara

Consolidation summaries mencatat mean IDSSE Jeremy0.598011, Ryu0.572393, Rasya0.520453, Farras0.333041. Empat angka ini tidak membuktikan urutan kualitas model terbaru:

- `farras_migration/predict.py` masih wave2, mengganti training menjadi dua IDSSE pertama yang bukan target (`[:2]`), bukan protokol Metrica asli atau wave5 lintas provider. README masih mengatakan scaffold pending walaupun summary tersedia. Folder remote Farras belum memuat wave3..5 lokal.
- `ryu_migration/predict.py` memakai shipped `weights_nometrica` untuk IDSSE yang sudah masuk27-match training. Bukan held-out checkpoint bagi tujuh match benchmark. Ada prior shift0.407 dan interpolasi25fps yang berbeda dengan adapter Farras.
- `jeremy_migration/predict.py` melatih `setseq_noaug`, sedangkan registry menyebut SetSeq boundary-aware terbaik. Ini adapter, bukan reproduksi kandidat terbaru exp7.
- `rasya_migration/predict.py` memakai touch context yang bergantung pada held-out GT untuk orientasi tim dan X mask.
- `common_benchmark/config.json` degrader hash masih `TBD`; runner tidak membaca config itu, menerima daftar match bebas dan default bootstrap0. `score_wrapper.py` mengambil label berdasar urutan, tidak memvalidasi kesamaan frame IDs kedua CSV. Membutuhkan manifest input/split/fit dan validasi alignment sebelum layak menjadi benchmark bersama.

Tidak ada common-benchmark rerun dalam wave6. Source branch consolidation tidak diubah diam-diam. Perbandingan wave6 memakai OOF/protokol lokal yang sama dengan wave5 dan scorer resmi, dengan kontrol yang harus persis mereproduksi skor lama.

### 3. Fitur nearest-opponent Ryu salah untuk pemain opposite-view

`origin/ryu:experiments/features.py:174-190`: mask kandidat `~same` berdasar view team, bukan tim pemain j. Pemain dari tim opposite-view masuk kandidat dirinya sendiri sehingga nearest distance0. Snapshot lokal `wave4/reference/features.py` identik pada bagian ini; learned features kita juga mewarisi behavior tersebut.

Perbaikannya harus membuat opponent mask relatif terhadap `T[j]`, mengeluarkan diri sendiri, lalu membangun ulang fitur dan melatih ulang semua fold/final checkpoint terkait. Mengganti hanya inference akan menimbulkan training/inference feature mismatch. Wave6 sengaja **tidak** menyebut bug ini sudah diperbaiki: seluruh probabilitas/checkpoint lama tetap frozen, efek fusion/decoder diuji terpisah.

Ryu juga mempunyai perbedaan calibration validasi versus shipped: `017/remote_pipeline.py` menyimpan match-weighted training rate; shipped `015/run.py` memakai source-balanced rate dan prior shift0.407. Validasi ulang harus menguji pipeline deployment yang sama, bukan primitive decoder saja. `--retrain` di015 hanya melatih3 Metrica, bukan mereproduksi shipped27-match model. TECHNICAL_DOCUMENT masih menggambarkan model/angka lama.

### 4. Rasya mempunyai unit bug dan held-out-label dependence

`origin/rasya:experiments/012_spellf1/barath.py:79-82` membagi ATD speed dengan3.6, padahal `DATA.md:85-86` menyatakan m/s. Cutoff slow2m/s efektif berubah menjadi7.2m/s pada input asli. Ryu017 NOTES melaporkan cue tersebut0.014 sebagaimana tertulis,0.400 setelah koreksi; angka ini laporan tim, tidak diulang di wave6.

`012_spellf1/touch.py:31-47` memakai full held-out GT agreement untuk membalik tim; `deployable.py:49-56` menyalin GT X mask. Mapping tim dan area graded harus berasal dari metadata sebelum menyebut inference bebas label. CONTEXT mencatat0.5263 dev/0.5057 confirm; longslow0.5504/0.5413 ditolak oleh adoption threshold internal. Confirm pair sudah dipakai berulang. TECHNICAL menggambarkan002 lama, bukan012; evidence vault dan gate.py tidak seluruhnya dilacak.

Pipeline Rasya tidak diimpor sebagai produksi wave6. Source teammate tetap utuh.

### 5. Jeremy exp7 berhasil mengurangi damage simulasi, belum lolos adopsi

`origin/jeremy:experiments/exp_7/REPORT.md`, `results/decision.json` dan `summary.json`: measured-view provider spellF10.677863, gain terhadap clean reference0.013727 dengan interval[0.003328,0.026526],6/7 wins. Tetapi ketiga klausul ctrl gagal: gain0.002216 dengan interval melintasi0; no-harm -0.005872 dan stress -0.005444 melewati batas -0.005. Tidak ada full-data model/candidate/rebuild exp7; jangan menganggapnya submission siap pakai.

Exp6 detector precision@2m0.760362/recall0.230371 gagal gate0.90/0.50. Measured recovery simulation sebelumnya menurunkan provider spellF1 sekitar0.125. Exp7 menggunakan test-derived detector noise/false-positive rates dan target-derived coverage/run lengths; bukan cross-source holdout independen. Touch-only exp4/5 membantu proxy, tetapi target dead share exp5 menjadi0.488370 versus prior0.407. Tidak menambahkan noisy ball recovery/touch ke wave6 tanpa validasi transfer.

Hash verification Jeremy exp7 memeriksa prereg/judge/constants tetapi tidak menegakkan seluruh dictionary source hashes; evaluation cache dilewati berdasarkan file exists. Ini celah reproduksi potensial, bukan bukti hasil dimanipulasi. Wave6 memverifikasi source/input hashes dan menolak missing probabilities, bukan memuat cache berdasarkan keberadaan saja.

### 6. Wave5 kita masih punya false-transition tradeoff

Wave5 mean spellF1 development0.670099 versus wave4 0.656490. Public spellF1 tetap0.772414, bukan private score. Short recall naik, tetapi pooled stress precision turun46.13% menjadi43.84%; recall GT>=15detik turun71.98% menjadi68.64%. Hanya2/6 paired stress menang. Holdout biasa sudah terekspos, bukan test baru.

Wave6 mengisolasi konservatif fusion/heads/occlusion costs/hysteresis/A-B-A guard dengan probabilitas frozen. Tidak ada training baru atau public tuning. Bootstrap memakai unique match sebagai cluster, tidak menganggap kondisi clean/stress sebagai match terpisah. Selection/public phases dipisah; kandidat lama tetap bila aturan promosi gagal.

## Batas perbandingan skor

Ryu017 melaporkan0.679 pada30 grouped-heldout matches, full-model public0.747; default nometrica public tidak teridentifikasi dari NOTES saja. Adapter lokal wave4 pernah diukur0.760, tetapi bukan reproduksi byte-exact source Ryu. Jeremy0.748 memakai L_ctrl yang menggabungkan short spells, bukan GT provider asli. Angka-angka tersebut tidak dapat dibandingkan langsung dengan equal-provider objective wave5/6.

Main metric resmi tetap mean spellF1 IoU0.3/0.4/0.5/0.6/0.7, seluruh possessions. Baseline resmi publik smooth50 sekitar0.325581; baseline full-match yang dipublikasikan0.2433. Public hanya17GT possessions. Private/full-match score model tim belum diketahui.

## Provenance

URL tiap source: `https://github.com/Ryu2804/more-than-a-hack-2027/blob/<commit>/<path>` memakai commit pada tabel. Source audit tidak memerlukan execute scripts dari branch lain. File wave1..5, main, serta eksperimen teammate tidak diubah. Update baru hanya wave6 dan indeks Farras. Laporan ini tidak memperbaiki remote registry atau mendorong branch otomatis.
