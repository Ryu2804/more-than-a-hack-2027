# Hasil temporal methods

## Keputusan

Kontrol tetap `short75% + SetSeq25% / dead45`, mean official spellF1 IDSSE **0.6154575861548992**. Primary yang ditetapkan sebelum evaluasi, `duration/strength0.5_bias0`, menghasilkan0.6058379034594431. Selisih -0.009619682695456211, paired-bootstrap95%CI[-0.019066773114166515,-0.002949900067270631], menang1/7. Tidak lolos aturan promosi; kontrol tidak diganti.

Best raw exploratory adalah `consensus/alpha0.75_contact0.5`: spellF1 **0.6169697856726231**, macroF1 0.909851793, naik0.0015121995177236971, menang5/7. Interval selisih[-0.00015497175080957811,0.0035096818464089275] masih melintasi0 dan gain kurang dari0.005. Ini peningkatan kecil yang belum kuat, bukan bukti model baru pasti lebih baik. Varian alpha0.5/contact0.5 menang6/7 dan interval nominal positif, tetapi gain0.0014639193262721006 juga kurang dari0.005 dan dipilih dari banyak varian exploratory.

Tidak ada skor public/private baru atau target submission. Ini adaptive research pada tujuh pertandingan yang sudah berulang kali dilihat, bukan fresh blind estimate. Interval tidak dikoreksi multiple comparisons atau riwayat pemilihan.

## Metode baru dan seluruh hasil

Semua memakai cached OOF neural predictions yang sama dan fixed75/25probability fusion. Duration priors saja belajar dari enam training matches masing-masing fold. Consensus mengurangi switch penalty di directional change yang disepakati kedua model, ditambah observed contact. Boundary relocation mengoptimalkan lokasi batas secara bersama sambil menjaga ordered state runs dan jumlahnya. HSMM memodelkan durasi H/A/D dengan unbounded geometric tail, tanpa hard dead gate atau target jumlah spell. Detail rumus dibekukan di [PLAN.md](PLAN.md).

| Konfigurasi | spellF1 | macroF1 | Selisih | Menang/7 |
|---|---:|---:|---:|---:|
| `consensus/alpha0.75_contact0.5` | 0.616970 | 0.909852 | +0.001512 | 5 |
| `consensus/alpha0.5_contact0.5` | 0.616922 | 0.909863 | +0.001464 | 6 |
| `consensus/alpha0.75_contact0` | 0.616689 | 0.909815 | +0.001231 | 4 |
| `control/dead45` | 0.615458 | 0.909587 | +0.000000 | 0 |
| `consensus/alpha0.25_contact0.5` | 0.615177 | 0.909588 | -0.000281 | 2 |
| `consensus/alpha0.5_contact0` | 0.614820 | 0.909644 | -0.000637 | 2 |
| `consensus/alpha0.25_contact0` | 0.614806 | 0.909572 | -0.000652 | 0 |
| `boundaries/radius2_penalty0` | 0.614800 | 0.909956 | -0.000657 | 2 |
| `boundaries/radius2_penalty0.05` | 0.614339 | 0.909967 | -0.001118 | 2 |
| `boundaries/radius5_penalty0` | 0.613149 | 0.909675 | -0.002309 | 1 |
| `duration/strength0.25_bias0` | 0.613077 | 0.909092 | -0.002381 | 1 |
| `boundaries/radius5_penalty0.05` | 0.613017 | 0.909686 | -0.002441 | 1 |
| `boundaries/radius10_penalty0` | 0.612572 | 0.909494 | -0.002886 | 1 |
| `boundaries/radius10_penalty0.05` | 0.612572 | 0.909490 | -0.002886 | 1 |
| `duration/strength0.25_bias-0.5` | 0.612076 | 0.909371 | -0.003381 | 2 |
| `duration/strength0.5_bias-0.5` | 0.609884 | 0.908987 | -0.005574 | 2 |
| `duration/strength0.5_bias0` | 0.605838 | 0.908576 | -0.009620 | 1 |
| `duration/strength1_bias-0.5` | 0.590225 | 0.907336 | -0.025233 | 0 |
| `duration/strength1_bias0` | 0.583344 | 0.906548 | -0.032113 | 0 |

Primary metric mean official spellF1 across IoU0.3/0.4/0.5/0.6/0.7, seluruh H/A spells termasuk pendek. MacroF1 secondary. Menang dihitung per unique match terhadap kontrol; ties bukan wins.

## Per pertandingan

| Match | Control | Primary HSMM | Best consensus |
|---|---:|---:|---:|
| J03WMX | 0.546667 | 0.543396 | 0.553431 |
| J03WN1 | 0.566628 | 0.563088 | 0.568009 |
| J03WOH | 0.651744 | 0.645655 | 0.654440 |
| J03WOY | 0.641802 | 0.604941 | 0.643000 |
| J03WPY | 0.618994 | 0.619799 | 0.617827 |
| J03WQQ | 0.654496 | 0.642735 | 0.655435 |
| J03WR9 | 0.627871 | 0.621250 | 0.626646 |

## Diagnosis sesudah run

Recall berikut memakai IoU0.75 sebagai diagnostic, bukan primary metric. Ground truth sama untuk semua metode. Jumlah prediksi adalah pooled H/A spells; count error adalah mean relative error per match.

| Metode | Prediksi/GT spells | Mean count error | Cocok<4s /1.621 | Cocok4 sampai<15s /1.058 | Cocok>=15s /382 |
|---|---:|---:|---:|---:|---:|
| Control | 2208/3061 | -0.273891 | 230 | 539 | 301 |
| Primary HSMM | 2050/3061 | -0.324976 | 203 | 513 | 303 |
| Best consensus | 2218/3061 | -0.270864 | 233 | 546 | 299 |
| Boundary radius2/penalty0 | 2208/3061 | -0.273891 | 222 | 542 | 301 |

Primary HSMM justru menghasilkan lebih sedikit spells dan recall pendek lebih rendah; durasi panjang sedikit membaik. [INFERENCE] Duration prior pada bobot ini menambah smoothing yang tidak sesuai dengan masalah under-segmentation. Seluruh HSMM strength yang diuji kalah dari kontrol, sehingga tidak menaikkan strength atau mengadopsinya berdasarkan hipotesis saja.

Boundary relocation mempertahankan jumlah spells sesuai rancangan, tetapi semua radius/penalty yang diuji menurunkan spellF1. [INFERENCE] Menaikkan frame log-probability dengan urutan label tetap belum cukup untuk memperbaiki objective IoU, terutama jika spell pendek sudah hilang. Consensus menambah10pooledspells dan sedikit recall pendek/menengah; ini perubahan terukur tetapi kecil.

## Verifikasi dan provenance

47semantic tests lulus, termasuk exhaustive tiny-sequence optimality untuk HSMM dan boundary DP, censored training intervals, unforced257-frame duration tail, half resets, contact agreement, immutable inputs dan frozen decoder parity pada constant weights. Smoke19settings pada96observasi nyata dari dua babak lulus tanpa metric evaluation.

Full evaluation19settings x7fold=133metric rows selesai29.39detik, di luar semantic tests/smoke, preregistration dan independent rebuild. Nol neural fits baru; tujuh supervised duration-prior fits. Ini decoder research dengan cached predictions, bukan perbandingan biaya training atau arsitektur baru.

Independent rebuild pada proses lain selesai dengan133decoded arrays identik dan tujuh duration priors yang di-fit ulang identik. Semua proses checks/evaluate/rebuild selesai exit0.

Source/PLAN/input manifest SHA256: `38829cd0f0f48a1ffffa2d4dd0cca57c14dcc7f684a95b58eed0e697b0076b75`. Predecessor decoded control harus byte-identical per match, bukan hanya sama skor. Hasil lengkap dan per-match bootstrap inputs: `../results/temporal_methods_results.json`; protocol/checks/rebuild proof dibagikan di prefix yang sama. Raw training-duration lists tetap di ignored `out/priors/`; tidak dipublikasikan.
