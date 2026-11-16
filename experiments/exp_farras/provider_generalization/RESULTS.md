# Hasil provider generalization

## Kesimpulan untuk risiko overfit publik

**Belum ada dasar untuk menjamin model bebas overfit.** Kode lama yang diinspeksi memilih kandidat dari external-development metrics lalu membuka public GT sebagai diagnostic setelah freeze. Itu bukan bukti skor publik yang sudah dilihat tidak pernah memengaruhi riset berikutnya. Semua 27 external matches juga sudah terekspos pada campaign terdahulu. Audit dan file/line evidence: [EXPOSURE.md](EXPOSURE.md).

Campaign baru memperketat proses: tidak mengakses competition-public/private labels; fresh source-provider-only training; match/provider disjoint per arah; label-free target inference; semua predictions committed sebelum skor target baru dihitung. Ini mengurangi jalur kebocoran dan score-driven iteration **dalam campaign ini**, bukan menghapus riwayat adaptasi atau menghasilkan fresh blind test.

Regularizer baru tidak memberi peningkatan generalisasi yang memenuhi aturan. Kontrol tetap dipertahankan. Penurunan dibanding angka IDSSE ensemble lama tidak langsung membuktikan public overfit: protokol baru memakai provider lain sebagai sumber training, model standalone, decoder source-rate-only dan label/coverage berbeda. Angka 0.624577 dari campaign lama bukan pembanding setara.

## Eksperimen dan keputusan primary

Train 7 IDSSE r0 → test 20 SkillCorner clean; train 20 SkillCorner clean → test 7 IDSSE r0. Fresh PP-TCN, 135.219 parameter/network, tiga seeds, 500 updates, batch 16, crop 384. Semua lima resep memakai original short loss dan 50% event sampling; normalisasi, event indices dan rate calibration hanya dari source training. Total **30 seed fits / 15.000 optimizer updates**. RDrop memakai dua stochastic forwards per update, sehingga compute tidak identik.

Primary `modality_dropout` menghasilkan equal-provider original spellF1 **0.5303783116699301**, versus kontrol **0.5408640368900999**. Selisih **-0.010485725220169892**, provider-stratified paired-bootstrap 95% CI **[-0.016208310154948818, -0.004824001196390026]**. Delta target IDSSE -0.02193845247146054, target SkillCorner +0.0009670020311207562. Gagal syarat gain ≥ 0.01, gagal guard provider minimum -0.02, dan interval tidak mendukung kenaikan. Tidak dipromosikan.

Best raw exploratory `rdrop` **0.5408823298349871**, hanya **+0.00001829294488716563** terhadap kontrol; CI [-0.0016194618724505068, 0.001493495988374656]. Praktis sama, bukan bukti improvement. Tidak mengganti primary dengan raw winner.

Bootstrap 5.000 draws, seed 20261120, paired match resampling terpisah di setiap target provider lalu provider diberi bobot sama. Tujuh IDSSE dan 20 SkillCorner bukan 27 equal-weight rows untuk objective utama. Semua interval nominal, tidak dikoreksi multiple comparisons/adaptive history. Gap diagnostics tidak memilih model pengganti.

## Semua hasil transfer

`original` berarti carrier yang sudah tersedia: IDSSE tetap r0 corruption, SkillCorner tetap clean preparation. `gap` menambah 35% mask pada ball rows yang sebelumnya terlihat, dengan span 10–60 samples dan seed fixed; X16 dan seluruh PP/ball features dihitung ulang. Sebanyak 54 conditions berasal dari 27 matches yang sama, bukan 54 independent matches. Main metric: official mean spellF1 across IoU 0.3/0.4/0.5/0.6/0.7, seluruh H/A spells termasuk pendek.

| Metode | SC→IDSSE original | IDSSE→SC original | SC→IDSSE gap | IDSSE→SC gap | Equal mean original | Equal mean gap |
|---|---:|---:|---:|---:|---:|---:|
| RDrop | 0.503927 | 0.577837 | 0.480469 | 0.560447 | 0.540882 | 0.520458 |
| Control | 0.505107 | 0.576621 | 0.479137 | 0.559069 | 0.540864 | 0.519103 |
| PP token dropout | 0.503630 | 0.572568 | 0.480268 | 0.560256 | 0.538099 | 0.520262 |
| Modality dropout | 0.483168 | 0.577588 | 0.462700 | 0.555926 | 0.530378 | 0.509313 |
| MixStyle | 0.493595 | 0.558371 | 0.469814 | 0.551363 | 0.525983 | 0.510589 |

| Metode | Original delta [95% CI] vs control | Gap delta [95% CI] vs control | Menang original/27 |
|---|---|---|---:|
| RDrop | +0.000018 [-0.001619, 0.001493] | +0.001355 [-0.002065, 0.004565] | 15 |
| Control | 0 [0, 0] | 0 [0, 0] | 0 |
| PP token dropout | -0.002765 [-0.007056, 0.000416] | +0.001159 [-0.001007, 0.003391] | 9 |
| Modality dropout | -0.010486 [-0.016208, -0.004824] | -0.009790 [-0.014294, -0.005668] | 9 |
| MixStyle | -0.014881 [-0.018800, -0.010874] | -0.008514 [-0.013077, -0.004344] | 1 |

Modality dropout sedikit membaik pada target SkillCorner tetapi turun pada target IDSSE. MixStyle turun pada kedua provider. PP dropout/RDrop menunjukkan point estimates gap sedikit lebih baik, tetapi interval masing-masing masih melintasi 0. Tidak ada alasan score-driven untuk mengadopsi salah satunya dari hasil ini.

[INFERENCE] Menjatuhkan salah satu player representation secara penuh dapat menghilangkan informasi yang tetap diperlukan pada provider target; mixing statistics juga dapat mengganggu feature geometry yang informatif. Eksperimen ini tidak mengisolasi mekanisme tersebut. Hasil negatif dicatat, bukan disembunyikan sebagai kegagalan implementasi.

## Statistik decoder source-only

| Training source | Matches training | Possession spells/menit source | Dead share source |
|---|---:|---:|---:|
| IDSSE | 7 | 4.581112 | 0.448819 |
| SkillCorner | 20 | 2.591973 | 0.466170 |

Tidak ada target possession-rate atau dead-share calibration. [INFERENCE] Perbedaan densitas spell/proxy-label antarsumber dapat memengaruhi decoder saat transfer. Tracking quality, provider-label definitions dan domain shift bercampur pada evaluasi ini; penurunan skor tidak dapat diatribusikan hanya ke overfit.

## Bukti read isolation dan prediksi sebelum scoring

1. Preparation menyalin external labels ke vault dan membuat 54 observation/feature carriers tanpa y/y25. Preparation dan registration menghitung input digests, bukan metric. Tidak ada provided-game carrier atau competition GT dibuka.
2. Lima belas real smoke seed fits memakai 7 source IDSSE matches, 1 update/seed. Smoke inference dan saved-model replay menguji satu SkillCorner match pada dua conditions, tanpa target labels atau metric.
3. Sepuluh full source-only ensembles dilatih dengan source/test providers terpisah. Fit menolak target-provider original labeled packs/features dan seluruh vault. Predict/replay menolak semua original labeled packs dan vault; hanya observation-only carriers yang dipakai.
4. Semua 270 full target probability arrays **dan** decoded arrays direproduksi identik dari sepuluh model tersimpan pada proses lain. Tidak melatih ulang untuk menyesuaikan skor.
5. Seal write-once mengikat **300 artifacts**: 270 prediction files, 10 checkpoints, 10 metadata, 10 replay proofs. Seluruh coverage dan SHA256 cocok sebelum scorer boleh membuka external target label vault. Test scoring baru berlangsung sesudah segel lengkap.
6. Runtime guard menolak deliberate public-GT probe sebelum byte dibaca. Probes terpisah dari forbidden reads tak disengaja. Trusted Python open policy bukan OS sandbox; native I/O, existing descriptors atau adversarial bypass tidak sepenuhnya dicakup. `successful_forbidden_reads=0` pada report hook adalah properti mekanisme tersebut, bukan audit keamanan independen.

Ini membuktikan proses yang dijalankan dan integritas artefak yang diperiksa, bukan bahwa model pasti tidak overfit atau akan membaik di private target match.

## Verifikasi, perbaikan pre-freeze dan waktu

Sebanyak 59 semantic tests selesai: **58 lulus, 1 skip**. Coverage mencakup math/gradients/masking untuk regularizers, actual subprocess read-denial, seal tampering/missing-artifact, label-bearing carrier rejection, seal-before-vault ordering dan provider-balanced bootstrap. Alasan rinci skip tidak tersimpan dalam aggregate runner; tidak ada tes dihapus atau numeric tolerance dilonggarkan.

Proses verifikasi terpisah selesai: **270 metric rows identik**, 300 sealed artifacts cocok, 270 probability/decoded replays identik dari 10 saved-model ensembles. Dari 50 runtime reports: **140 intentional denied probes, 0 unexpected forbidden-read attempts**. Ini bukti read policy dan reproduksi yang dijalankan, bukan jaminan bebas overfit.

Sebelum manifest/full fits, initial checks menemukan relative-path fixture melintasi drive Windows dan floating-point cancellation pada MixStyle float32. Fixture dipindah ke temporary directory pada drive cwd yang sama; MixStyle mixing arithmetic dibuat float64 dan output tetap input dtype. Source enumeration juga diperbaiki agar tidak mencoba hash synthetic `_classes.py` milik PyTorch sebagai file nyata. Semua perbaikan sebelum source freeze dan sebelum full training; tidak mengganti konfigurasi setelah skor terlihat.

Sum 10 fit-job timers **819.110 detik**, termasuk source loading, fitting dan checkpoint/metadata preparation setelah input verification. Tidak termasuk preparation, final check 351.5 detik, startup/source verification, initial target inference/replay, seal/scoring/recompute atau orchestration overhead. Independent metric verification selesai dalam 159.30 detik. CUDA GPU jobs berurutan. Tidak mengklaim equal FLOPs karena RDrop dua forwards dan MixStyle memakai mixing arithmetic float64.

Source/PLAN/input manifest SHA256: `6ac1740c290d3ab332b33eb7a0a1ab6452c23bd6b285835cd1c3ba1c84039279`.
Prediction-seal SHA256: `6429e7462a2faf3353a914c29e99ed160aaf953d99e08fb267d40021474272c2`.

Aggregate JSON di sibling `results/provider_generalization_*.json`. Observation carriers, label vault, tracking, weights, predictions, mask spans dan raw seal tetap di ignored cache/out. Tidak ada public/private score baru, target CSV, commit atau push otomatis. Perintah dan batas pipeline: [README.md](README.md).
