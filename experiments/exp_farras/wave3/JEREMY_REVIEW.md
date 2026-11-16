# Audit branch Jeremy

Sumber dipin: [`0e35aae54e09ca744441681fb076f656bbc88b5a`](https://github.com/Ryu2804/more-than-a-hack-2027/tree/0e35aae54e09ca744441681fb076f656bbc88b5a/experiments/exp_1). Angka di bagian ini berasal dari artefak branch, belum reproduksi lokal checkpoint asli.

## Hasil yang dilaporkan

- `results/public_sanity.json`: spellF1 0.581818, macroF1 0.725033, 16 predicted possessions versus17 GT; interval bootstrap spellF1[0.40,0.751515].
- `REPORT.md`: proxy7 pertandingan IDSSE mean spellF1 sekitar0.656 dan macroF1 sekitar0.918. Pada proxy yang sama reference Ryu zero-shot sekitar0.487.
- `results/followup_decision.json`: centered boundary follow-up tidak lolos seluruh tiga aturan adopsi; tidak membuat submission follow-up.
- `results/target_summary.json`: decoder temperature1.302867, lambda_team0.5, lambda_dead0.5, gamma0;437 target predicted possessions; valid format dan hash reproduksi dicatat Jeremy.

Dibanding diagnostic Farras0.56, skor public Jeremy lebih tinggi0.021818. Window hanya17 GT possession dan interval Jeremy lebar. Ini tidak membuktikan keunggulan private/full-match atau signifikansi selisih kedua metode.

## Apa yang dilakukan

1. Anonymous player tokens: posisi, kecepatan turunan, tanda tim, offset ke bola, proximity, nearest marker; tidak memakai ID pemain.
2. DeepSets-style shared MLP, mean/max pooling per tim dan attention pooling keseluruhan; digabung12 global ball/gap/team features.
3. Dua-layer bidirectional GRU pada jendela60 detik, overlapping windows dengan tent weighting saat inference.
4. State head H/A/D dan auxiliary boundary head. Default width64, hidden64, training14epoch, batch32, coverage2, seed20261108, AdamW dan OneCycleLR.
5. Tiga corruption rounds: bola hilang sekitar41%, player gaps, tracklet permutations per tim, duplicate detections. Transformasi observasi tidak membaca label.
6. Augmentasi pertukaran H/A + rotasi180derajat, label H/A ikut ditukar; dead/unknown tetap.
7. Nested leave-one-match-out pada7 IDSSE:21 pair models dan7 outer models; decoder dipilih memakai validation model yang tidak melihat outer test maupun validation match.
8. Objective proxy spellF1 dikurangi penalti count error di atas30%; bootstrap berpasangan per match.

## Bagian yang didukung dan tidak

Laporan Jeremy mendukung augmentation (+0.055 proxy), model set-temporal dibanding LightGBM (+0.198), serta decoder dibanding argmax (+0.032). Boundary-aware decoding awal tidak pernah dipilih; gamma0 dipilih di semua fold. Forced boundary cost justru sekitar0.007 lebih rendah.

Centered follow-up memperkecil masalah boundary cost yang hanya menambah penalti switch, tetapi delta0.001277 berinterval95%[-0.000930,0.003820]; belum cukup untuk diadopsi. Count gate tetap gagal pada salah satu pertandingan, sehingga menambah smoothing bukan keputusan otomatis yang aman.

## Batas perbandingan

Model Jeremy dilatih dan dievaluasi pada degraded IDSSE, sedangkan Ryu reference zero-shot dan Farras sebelumnya dilatih pada proxy event Metrica. Angka proxy lintas campaign tidak langsung dibandingkan.

Target raw ATD metadata/text, model weights Jeremy, dan cache IDSSE tidak ikut commit. Hardware lokal memiliki CUDA; data IDSSE publik bisa diunduh dari mirror PySport. Campaign baru melatih adaptasi sendiri, tidak berpura-pura memakai checkpoint Jeremy atau mereproduksi persis0.581818.

## Kombinasi yang layak diuji

Hybrid global features menyisipkan7 bukti kontrol Farras ke model set-temporal. Campuran probabilitas menguji apakah tree kontrol membantu saat bola terlihat dan temporal model membantu selama gap. Feature-only GRU memisahkan manfaat konteks temporal dari encoder set. Decoder low/asymmetric cost menguji under-segmentation; centered boundary menjadi satu ablasi terbatas, bukan pencarian ulang besar.

Desain, split, dan aturan pemilihan: [PLAN.md](PLAN.md).
