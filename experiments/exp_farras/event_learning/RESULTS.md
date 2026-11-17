# Hasil event learning

## Keputusan

Best exploratory baru **`contact_replace/consensus`: mean official spellF1 IDSSE 0.6245765534033564**, macroF1 0.9115530617305891. Pembanding kuat `event_control/consensus` tetap 0.619562844991005. Gain **0.005013708412351555**, menang5/7, paired match-bootstrap95%CI **[-0.0002117957432191992,0.011265809452937545]**. Gain dan win count melewati dua batas, tetapi interval masih melintasi0. Ini juga bukan primary yang ditetapkan sebelum run, sehingga tidak dipromosikan otomatis.

Primary **`balanced_replace/consensus` mencapai 0.6191920412910819**: delta -0.00037080369992297565, CI[-0.0033139818714818287,0.002354190193876774], menang3/7 dan tie1. Primary gagal; strong research control tetap dipertahankan. Batas keputusan tidak diubah sesudah melihat hasil.

Alternatif exploratory `ranking_replace/dead45` mencapai0.6215116612127324, menang **7/7 terhadap strong consensus control**, delta0.0019488162217274738, CI[0.0007000594338442669,0.003537213935308925]. Gain kurang dari0.005 dan bukan primary. Saat dibandingkan event control dengan decoder dead45 yang sama, delta0.0025705249399012547, CI[-0.0003631786146264327,0.005429331168577739], menang5/7. Jadi hasil7/7 adalah perbandingan pipeline lengkap, bukan bukti efek ranking loss terisolasi dengan decoder identik.

Best versus legacy ensemble0.6154575861548992 mempunyai gain0.00911896724845713, CI[0.0023922816245338916,0.01569762203059722], menang6/7. Perbandingan lebih mudah ini tidak dipakai untuk menggantikan syarat terhadap strong control.

Semua interval memakai5.000paired bootstrap draws, seed20261118, unit unique match. Tujuh pertandingan, hipotesis dan kontrol sebelumnya sudah terekspos; interval nominal tidak mengoreksi adaptive history atau multiple comparisons. Ini riset IDSSE, bukan fresh blind test atau skor public/private kompetisi. Tidak ada target evaluation atau submission baru.

## Enam resep yang dijalankan

Semua memakai event sampler50%, tiga seeds0/1/2,500updates/seed,batch16,crop384, tujuh LOSO folds dan enam training matches lain. Total **126seedfits baru /63.000updates**. Tidak melatih pada heldout match atau memuat predecessor weights. Short/SetSeq/previous-event probabilities hanya dipakai dari authenticated OOF caches.

| Resep | Perubahan dari previous event TCN | Parameter/network |
|---|---|---:|
| balanced | Complete-spell inverse-sqrt team loss, dead BCE biasa | 135219 |
| focal | Conditional-team focal gamma2, original dead loss | 135219 |
| boundary | Auxiliary radius2 boundary target/BCE, head ketiga | 135284 |
| ranking | Signed adjacent H/A-change margin loss | 135219 |
| ema | EMA0.99 setelah setiap optimizer update; shadow untuk inference | 135219 |
| contact | Tujuh observed raw25fps contact/memory/dead-cue channels | 135667 |

Boundary target dihitung pada full training half sebelum crop, tidak melintasi unknown atau babak. Hanya frame pertama setiap half invalid, bukan seluruh half. Auxiliary head tidak dipakai decoder. EMA tidak memilih checkpoint berdasarkan validasi. Contact memakai cached two-view features yang sudah diaudit, tanpa GT orientation/D/X mask; seluruh tujuh channel ikut ball dropout mask. Complete-spell balanced weights dan original short weights mempunyai aturan censoring berbeda yang sengaja dipertahankan.

Update budgets sama, tetapi input/parameter/compute tidak identik untuk semua resep. Boundary mengubah head, contact menambah input dan EMA memakai shadow copy selama training. Detail rumus serta keputusan dibekukan di [PLAN.md](PLAN.md).

## Semua40konfigurasi

`replace`: short50% + SetSeq25% + modelbaru25%. `split`: short50% + SetSeq25% + previous-event12.5% + modelbaru12.5%. `pure`: modelbaru saja. Dua decoder tetap: dead45 dan consensus(alpha0.5/contact_beta0.5), tanpa retuning. Delta dan wins terhadap strong research control, bukan legacy.

| Konfigurasi | spellF1 | macroF1 | Delta strong control | Menang/7 |
|---|---:|---:|---:|---:|
| `contact_replace/consensus` | 0.624577 | 0.911553 | +0.005014 | 5 |
| `contact_replace/dead45` | 0.623893 | 0.911369 | +0.004330 | 4 |
| `ranking_replace/consensus` | 0.621600 | 0.910293 | +0.002037 | 4 |
| `ranking_replace/dead45` | 0.621512 | 0.910192 | +0.001949 | 7 |
| `contact_split/consensus` | 0.620972 | 0.910710 | +0.001409 | 4 |
| `boundary_split/consensus` | 0.620675 | 0.910270 | +0.001112 | 4 |
| `ranking_split/consensus` | 0.620519 | 0.910168 | +0.000956 | 6 |
| `contact_split/dead45` | 0.620461 | 0.910488 | +0.000898 | 3 |
| `balanced_split/consensus` | 0.620023 | 0.910201 | +0.000460 | 5 |
| `boundary_replace/consensus` | 0.620014 | 0.910291 | +0.000451 | 3 |
| `boundary_split/dead45` | 0.620003 | 0.910203 | +0.000440 | 4 |
| `boundary_replace/dead45` | 0.619767 | 0.910271 | +0.000204 | 4 |
| `event_control/consensus` | 0.619563 | 0.909983 | +0.000000 | 0 |
| `ranking_split/dead45` | 0.619542 | 0.910006 | -0.000021 | 3 |
| `contact_pure/consensus` | 0.619249 | 0.907418 | -0.000313 | 4 |
| `balanced_split/dead45` | 0.619238 | 0.910038 | -0.000325 | 4 |
| `balanced_replace/consensus` | 0.619192 | 0.910290 | -0.000371 | 3 |
| `event_control/dead45` | 0.618941 | 0.909976 | -0.000622 | 3 |
| `ema_replace/consensus` | 0.618872 | 0.909735 | -0.000691 | 3 |
| `ema_split/consensus` | 0.618762 | 0.909831 | -0.000801 | 2 |
| `ema_split/dead45` | 0.618524 | 0.909732 | -0.001039 | 2 |
| `balanced_replace/dead45` | 0.618348 | 0.910172 | -0.001215 | 2 |
| `ema_replace/dead45` | 0.618150 | 0.909687 | -0.001413 | 2 |
| `contact_pure/dead45` | 0.617985 | 0.907221 | -0.001578 | 4 |
| `focal_split/consensus` | 0.617877 | 0.910107 | -0.001686 | 4 |
| `legacy/consensus` | 0.616922 | 0.909863 | -0.002641 | 3 |
| `focal_split/dead45` | 0.616871 | 0.910046 | -0.002691 | 3 |
| `legacy/dead45` | 0.615458 | 0.909587 | -0.004105 | 2 |
| `balanced_pure/consensus` | 0.615189 | 0.903583 | -0.004373 | 2 |
| `balanced_pure/dead45` | 0.614202 | 0.903348 | -0.005361 | 2 |
| `focal_replace/consensus` | 0.613417 | 0.909840 | -0.006146 | 3 |
| `focal_replace/dead45` | 0.612647 | 0.909697 | -0.006916 | 2 |
| `ranking_pure/consensus` | 0.611603 | 0.904330 | -0.007960 | 2 |
| `ranking_pure/dead45` | 0.610870 | 0.904315 | -0.008693 | 1 |
| `boundary_pure/consensus` | 0.608296 | 0.904209 | -0.011267 | 1 |
| `boundary_pure/dead45` | 0.608225 | 0.904188 | -0.011338 | 1 |
| `ema_pure/consensus` | 0.605611 | 0.903409 | -0.013951 | 1 |
| `ema_pure/dead45` | 0.605592 | 0.903368 | -0.013971 | 0 |
| `focal_pure/consensus` | 0.582393 | 0.901827 | -0.037170 | 0 |
| `focal_pure/dead45` | 0.580623 | 0.901463 | -0.038940 | 0 |

Primary metric tetap official mean spellF1 across IoU thresholds0.3/0.4/0.5/0.6/0.7, seluruh H/Aspells termasuk pendek. MacroF1 secondary. Semua280configuration-match rows memakai label, masking unknown dan expansion25fps yang sama.

## Secondary standalone contrasts

Perbandingan exploratory sesudah run, tidak mengubah keputusan primary. Semua memakai dead45 dan dibandingkan previous event standalone0.6080225797645553, bukan ensemble kontrol. Parameter/input parity berbeda untuk boundary/contact sebagaimana di atas. Ini menilai resep training secara lebih langsung daripada membandingkan pure model dengan ensemble kuat.

| Resep standalone | Delta previous event | Paired95%CI delta | Menang/7 |
|---|---:|---|---:|
| balanced | +0.006179 | [-0.001047,0.014312] | 5 |
| focal | -0.027400 | [-0.041041,-0.013547] | 0 |
| boundary | +0.000203 | [-0.005157,0.005124] | 3 |
| ranking | +0.002848 | [-0.000978,0.007167] | 5 |
| ema | -0.002430 | [-0.006575,0.001624] | 2 |
| contact | +0.009963 | [-0.006913,0.026013] | 5 |

Contact memberi mean tertinggi, tetapi variasi antar match masih besar. Focal turun di semua match pada kontrast standalone; EMA0.99 tidak mengalahkan previous event pada mean. Boundary auxiliary nyaris netral. Balanced meningkatkan standalone point estimate, tetapi tidak otomatis memperbaiki ensemble primary. [INFERENCE] Keuntungan standalone dan komplementaritas terhadap short/SetSeq berbeda; menumpuk dua recipe yang pernah membaik tidak menjamin ensemble membaik.

## Per pertandingan

| Match | Strong control | Primary balanced | Best contact | Ranking/dead45 |
|---|---:|---:|---:|---:|
| J03WMX | 0.551825 | 0.554070 | 0.557959 | 0.555152 |
| J03WN1 | 0.569072 | 0.574570 | 0.590607 | 0.575199 |
| J03WOH | 0.663372 | 0.659302 | 0.664723 | 0.664151 |
| J03WOY | 0.640699 | 0.633540 | 0.635705 | 0.641884 |
| J03WPY | 0.618333 | 0.616874 | 0.625313 | 0.619945 |
| J03WQQ | 0.663243 | 0.665591 | 0.668299 | 0.663599 |
| J03WR9 | 0.630395 | 0.630395 | 0.629429 | 0.630653 |

## Diagnostic possession pendek

IoU0.75di sini diagnostic saja, bukan primary. Semua3.061GT H/Aspells tetap dipertahankan. Predicted/matched spell counts pooled; count error mean per match.

| Pipeline | Prediksi/GT spells | Mean count error | Cocok<4s /1.621 | Cocok4 sampai<15s /1.058 | Cocok>=15s /382 |
|---|---:|---:|---:|---:|---:|
| Strong event control | 2241/3061 | -0.263341 | 243 | 542 | 300 |
| Primary balanced | 2253/3061 | -0.259366 | 248 | 550 | 300 |
| Best contact | 2242/3061 | -0.262804 | 263 | 546 | 304 |
| Ranking replace/dead45 | 2259/3061 | -0.257845 | 247 | 547 | 300 |

Best contact menambah20matchedshortspells dengan jumlah prediksi hanya bertambah1. [INFERENCE] Sinyal ini lebih sesuai dengan perbaikan penetapan tim/timing daripada sekadar memaksa jumlah spell lebih banyak. Itu belum membuktikan generalisasi pada pertandingan baru; selisih best versus strong control masih mempunyai interval yang melintasi0.

## Verifikasi dan publication boundary

- 77 semantic tests lulus: numerical/gradient loss references, unknown masks, full-half boundary/crop semantics, contact batch/view/mask parity, EMA mutation/buffers/serialization, exact fold/seed/budget/probability/replay rejection.
- 18 real smoke seedfits, masing-masing1update, enam fullheldout checkpoint replays identik. Smoke tidak dicampur ke performance estimate500updates.
- Sebelum full fits,28predecessor control decoded arrays direproduksi byte-identical; dua probability controls, dua decoders, tujuh matches.
- 42 full model-family/fold jobs selesai,126actualseedfits/63.000updates. Actual training names, seed order, update counts, EMA update counts dan per-family parameter counts divalidasi.
- 42 saved-model ensembles dimuat ulang di proses terpisah dan menghasilkan fullheldout probability arrays identik. Semua proses fit/replay/evaluate/rebuild exit0.
- Independent rebuild pada proses lain: **280decoded arrays identik**, predecessor controls tetap byte-identical. Ini inference/decode reproduction, bukan full training refit kedua.
- Sum42job timers setelah manifest verification: **3.685,890detik**, mencakup data loading, fitting, initial inference dan penyimpanan. Tidak termasuk startup/source verification, separate replay processes, semantic/smoke checks, final evaluate/rebuild atau orchestration overhead. GPUCUDA berurutan.

Manifest SHA256: `14327721dbf19145c288c54fd95f78564ec427b64d284f070ba1788f8d6095e1`. Source/PLAN/input/provenance dibekukan sebelum full fits, tidak direvisi agar sesuai skor baru.

Aggregate exports: `../results/event_learning_results.json`, `event_learning_protocol.json`, `event_learning_checks.json`, `event_learning_verification.json`, `event_learning_training.json`, `event_learning_comparisons.json`. Raw tracking, labels, weights, probabilities dan per-frame targets tetap ignored di out/. Tidak ada public/private score baru, targetCSV, commit atau push otomatis. Perintah reproduksi dan sumber komponen: [README.md](README.md).
