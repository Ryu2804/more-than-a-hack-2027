# Temporal methods

Tiga metode baru di atas ensemble short-loss TCN75% + SetSeq25%: explicit-duration HSMM, consensus change-point, dan ordered boundary relocation. Campaign lama dan folder eksperimen anggota lain tidak diubah. [PLAN.md](PLAN.md) membekukan hipotesis, konfigurasi dan aturan keputusan sebelum evaluasi baru.

## Menjalankan

Dari root repo, Python3.10 dengan environment fair_idsse/cross_branch. Perlu original r0 packs, cached OOF short_tcn/setseq dari fair_idsse, controls25fps dan provenance campaign cross_branch yang sudah lengkap. Tidak membuka weights lama atau melatih ulang jaringan.

```powershell
python -B experiments/exp_farras/temporal_methods/check.py
python -B experiments/exp_farras/temporal_methods/run.py all
python -B experiments/exp_farras/temporal_methods/run.py rebuild
```

`check.py` menjalankan semantic tests, termasuk exhaustive tiny-path proofs untuk dua dynamic programs, lalu smoke semua19settings pada96observasi nyata dari dua babak. Smoke tidak menghitung metric. `all` memverifikasi predecessor, membekukan source/PLAN/input hashes, fit duration priors dari enam training matches tiap fold dan mengevaluasi133setting-match pairs. Hasil selesai tidak ditimpa; campaign baru harus memakai folder baru. `rebuild` pada proses terpisah refit tujuh duration priors dan membangun ulang133decoded arrays dari inputs yang dibekukan.

## Pembacaan hasil

Baseline adalah hasil exploratory terbaik sebelumnya, mean official spellF1 IDSSE0.6154575861548992. Primary baru: HSMM strength0.5/bias0.0. Aturan promosi: gain>=0.005, menang>=5/7matches dan paired match-bootstrap95%lower>0. Semua18varian baru dilaporkan, tetapi raw winner tidak otomatis mengganti primary yang gagal. Semua match pernah terekspos; bootstrap tidak mengoreksi adaptive history atau multiple comparisons. Bukan fresh blind test.

Primary metric mempertahankan semua H/A spells dan mean IoU0.3/0.4/0.5/0.6/0.7 lewat official scorer. Diagnostic recall IoU0.75 per durasi hanya menjelaskan error; bukan metric utama. Semua sumber memakai fold, expansion25fps dan masking unknown yang sama. Duration priors belajar raw labels dari enam training matches, bukan prediksi OOF fold lain yang bisa melibatkan heldout. Tidak ada held-out label calibration atau oracle orientation/D/X mask.

Hasil dan keputusan: [RESULTS.md](RESULTS.md). Aggregate JSON dibagikan di `../results/temporal_methods_*.json`. Raw priors, decoded hashes dan laporan kerja tersimpan di ignored `out/`. Tidak membuat target submission, evaluasi public baru, commit atau push otomatis.

## Referensi

[Yu2010: Hidden semi-Markov models](https://www.sciencedirect.com/science/article/pii/S0004370209001416) untuk explicit-duration modeling; [Ishikawa2021: boundary-aware temporal segmentation](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html) untuk motivasi boundary-focused decoding. Implementasi di sini adaptasi sendiri, bukan reproduksi arsitektur atau hasil paper. Provenance model/data: [fair_idsse](../fair_idsse/README.md) dan [cross_branch](../cross_branch/SOURCES.md).
