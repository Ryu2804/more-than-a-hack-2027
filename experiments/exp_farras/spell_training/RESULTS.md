# Hasil spell training

## Keputusan

Primary `event_replace25/dead45` mencapai mean official spellF1 IDSSE **0.6189411362728311**, versus kontrol0.6154575861548992. Gain0.0034835501179317934, paired match-bootstrap95%CI[0.0006699738275977982,0.006503930284240906], menang5/7. Sinyal selisih positif, tetapi gain belum mencapai syarat0.005 yang dibekukan sebelum run. **Primary tidak dipromosikan; kontrol tetap dipertahankan.** Aturan tidak diubah setelah melihat skor.

Best raw exploratory: `event_replace25/consensus`, spellF1 **0.619562844991005**, macroF1 **0.9099829750309686**. Gain0.004105258836105574 terhadap kontrol dead45, CI[0.0010056306129022005,0.007612324713658802], menang5/7. Gain juga kurang dari0.005, dan konfigurasi ini bukan primary.

Best sebelumnya0.6169697856726231 merupakan exploratory consensus varian lain; best raw baru lebih tinggi0.0025930593183819406. Untuk mengisolasi efek model dari decoder, best baru dibandingkan kontrol dengan consensus yang sama(alpha0.5/contact0.5): gain0.0026413395098334735, CI[-0.0005110195422673357,0.00601190972269252], menang4/7. Interval ini masih melintasi0. Jangan menyebut seluruh kenaikan terhadap dead45 sebagai efek training baru saja.

Semua interval memakai5,000paired bootstrap draws, seed20261117, unit unique match. Tujuh pertandingan dan hipotesis sudah terekspos sebelumnya; interval nominal tidak mengoreksi adaptive history atau multiple comparisons. Ini hasil riset IDSSE, bukan skor public/private kompetisi. Tidak ada evaluasi target atau submission baru.

## Apa yang berubah

- Spell-balanced TCN: team BCE berbobot inverse-square-root durasi complete H/Aspells, normalized oleh weighted live mass; dead BCE biasa. Run yang menyentuh half endpoint atau unknown tidak dianggap complete.
- Event TCN: original short-loss tetap, tetapi50%kesempatan mengganti crop dalam match yang sudah dipilih dengan crop sekitar complete H/Aspell<4detik. Jitter/clamp menjaga satu babak dan event tetap masuk crop. Pemilihan match tetap seimbang; label heldout tidak dipakai.
- Dual-scale TCN: gabungkan Conv5dilation asli dan dilation1dengan kernel/bias yang sama sebelum GELU. Parameter tidak bertambah; convolution compute bertambah. Loss dan sampling tetap original short recipe.

Ketiganya training fresh pada enam match lain di setiap LOSO fold,3seeds/500updates/batch16/crop384. Sama29global channels, PPfeatures48x10, width64 dan **135.219parameter per network**. Total63seedfits/31.500updates. Tidak memuat predecessor weights. SetSeq dan short-loss control probabilities hanya reused dari audited heldout cache, bukan ditrain ulang.

## Seluruh konfigurasi

Semua angka mean per-match tujuh IDSSE folds. `replace25` berarti short50% + SetSeq25% + modelbaru25%; `base75` berarti75%dari kontrol probability mixture ditambah25%modelbaru. `pure` berarti modelbaru saja. Consensus memakai fixed alpha0.5/contact0.5, bukan mencari ulang best decoder. Selisih terhadap `control/dead45`.

| Konfigurasi | spellF1 | macroF1 | Delta | Menang/7 |
|---|---:|---:|---:|---:|
| `event_replace25/consensus` | 0.619563 | 0.909983 | +0.004105 | 5 |
| `event_replace25/dead45` | 0.618941 | 0.909976 | +0.003484 | 5 |
| `control/consensus` | 0.616922 | 0.909863 | +0.001464 | 6 |
| `control/dead45` | 0.615458 | 0.909587 | +0.000000 | 0 |
| `spell_replace25/dead45` | 0.615418 | 0.909749 | -0.000040 | 4 |
| `spell_replace25/consensus` | 0.615025 | 0.909745 | -0.000433 | 3 |
| `event_base75/consensus` | 0.615014 | 0.908300 | -0.000443 | 5 |
| `dual_replace25/consensus` | 0.613749 | 0.909313 | -0.001709 | 3 |
| `event_base75/dead45` | 0.613662 | 0.908173 | -0.001795 | 3 |
| `spell_base75/consensus` | 0.613384 | 0.908575 | -0.002074 | 2 |
| `dual_replace25/dead45` | 0.612194 | 0.909044 | -0.003263 | 1 |
| `spell_base75/dead45` | 0.611257 | 0.908367 | -0.004200 | 0 |
| `dual_base75/consensus` | 0.608949 | 0.907944 | -0.006509 | 0 |
| `event_pure/consensus` | 0.608846 | 0.904182 | -0.006611 | 2 |
| `event_pure/dead45` | 0.608023 | 0.904088 | -0.007435 | 2 |
| `dual_base75/dead45` | 0.607290 | 0.907672 | -0.008168 | 0 |
| `spell_pure/dead45` | 0.604291 | 0.902495 | -0.011167 | 0 |
| `spell_pure/consensus` | 0.604094 | 0.902565 | -0.011363 | 1 |
| `dual_pure/dead45` | 0.587528 | 0.900496 | -0.027930 | 0 |
| `dual_pure/consensus` | 0.587513 | 0.900674 | -0.027944 | 0 |

## Ablation model tanpa ensemble

Perbandingan secondary berikut dihitung sesudah run dan tidak mengubah keputusan primary. Pembanding adalah original short-loss PP-TCN/dead45 dari fair_idsse, **bukan** kontrol ensemble0.615458. Fold, input, decoder, seeds dan update budget sebanding. Ini menunjukkan efek training yang bisa tertutup bila semua pure models hanya dibandingkan ensemble.

| Model standalone/dead45 | spellF1 | Delta dari original short-TCN | Paired95%CI delta | Menang/7 |
|---|---:|---:|---|---:|
| Original short-loss TCN | 0.595755 | 0 | [0,0] | 0 |
| Spell-balanced TCN | 0.604291 | +0.008535 | [0.003403,0.012540] | 6 |
| Event-sampled TCN | 0.608023 | +0.012267 | [0.006049,0.017750] | 6 |
| Dual-scale TCN | 0.587528 | -0.008228 | [-0.014495,-0.003203] | 1 |

Spell-balanced loss dan event sampling memperbaiki standalone TCN, tetapi belum menjadi pengganti ensemble kontrol. Dual-scale turun bahkan terhadap original standalone; tidak diadopsi. Best ensemble memakai event TCN bersama short lama dan SetSeq, bukan membuang kedua kontrol.

## Per pertandingan

| Match | Control | Primary event | Best event + consensus |
|---|---:|---:|---:|
| J03WMX | 0.546667 | 0.551154 | 0.551825 |
| J03WN1 | 0.566628 | 0.568349 | 0.569072 |
| J03WOH | 0.651744 | 0.655977 | 0.663372 |
| J03WOY | 0.641802 | 0.640699 | 0.640699 |
| J03WPY | 0.618994 | 0.617222 | 0.618333 |
| J03WQQ | 0.654496 | 0.665041 | 0.663243 |
| J03WR9 | 0.627871 | 0.634146 | 0.630395 |

## Diagnosis possession pendek

Ini IoU0.75diagnostic, bukan primary metric resmi yang tetap mean IoU0.3/0.4/0.5/0.6/0.7. Semua3.061GT H/Aspells dipertahankan. Count error mean per-match; spell counts dan matched counts pooled.

| Metode | Prediksi/GT spells | Mean count error | Cocok<4s /1.621 | Cocok4 sampai<15s /1.058 | Cocok>=15s /382 |
|---|---:|---:|---:|---:|---:|
| Control | 2208/3061 | -0.273891 | 230 | 539 | 301 |
| Primary event ensemble | 2235/3061 | -0.265543 | 239 | 539 | 300 |
| Best event + consensus | 2241/3061 | -0.263341 | 243 | 542 | 300 |
| Event standalone/dead45 | 2389/3061 | -0.214072 | 229 | 534 | 287 |
| Spell standalone/dead45 | 2426/3061 | -0.201483 | 215 | 540 | 285 |
| Dual standalone/dead45 | 2293/3061 | -0.245651 | 186 | 503 | 290 |

Best ensemble menambah33predictedspells dan13matchedshortspells terhadap kontrol. [INFERENCE] Pengayaan crop membantu informasi possession pendek, tetapi angka event standalone menunjukkan menaikkan jumlah spells saja tidak menjamin kecocokan IoU. Kenaikan best ensemble kecil dan bervariasi antar pertandingan.

Jumlah complete short events per training record berikut dihitung sebelum sampling. Tiap fold hanya memakai enam match selain heldout.

| Match | Eligible short events |
|---|---:|
| J03WMX | 278 |
| J03WN1 | 284 |
| J03WOH | 196 |
| J03WOY | 249 |
| J03WPY | 206 |
| J03WQQ | 178 |
| J03WR9 | 170 |

Tidak ada match tanpa event pada campaign ini; perilaku mempertahankan uniform window untuk match tanpa event diuji terpisah.

## Verifikasi dan artefak

- 27 semantic tests lulus: full-run duration/censoring, unknown gradients dan weighted denominator, view symmetry, event enrichment/source-match balance/half safety, original RNG/batch parity pada non-event sampling, dual-scale numerical/gradient proof dan PP mask/permutation behavior.
- 9 smoke seed fits masing-masing 1 update pada real training data, tiga full heldout inference replays identik. Smoke tidak dicampur ke performance estimate 500 updates.
- 21 full training jobs, 63 seed fits, 31.500 actual updates; metadata mencatat six-match membership dan setiap seed/parameter count. Semua proses exit0.
- 21 saved-model ensembles dimuat ulang di proses terpisah; seluruh full heldout probability arrays identik dengan cached outputs.
- Independent decoder rebuild: **140 arrays identik**, kontrol predecessor byte-identical. Ini inference/decode reproduction, bukan full training refit.
- Sum job durations **1.816,565 detik** mencakup loading, training, initial inference dan penyimpanan; tidak termasuk separate replay subprocesses, registration, tests, final evaluation/rebuild atau orchestration overhead. GPU CUDA dipakai, pekerjaan GPU berurutan.

Manifest SHA256: `2db5b8e793cfd43559425d600aa282422558e135016178944b3c8bfaf3d306ca`. Sumber dan [PLAN.md](PLAN.md) dibekukan setelah smoke sebelum full fits, tidak diperbarui agar cocok dengan hasil.

Aggregate exports: `../results/spell_training_results.json`, `spell_training_protocol.json`, `spell_training_checks.json`, `spell_training_verification.json`, `spell_training_training.json`, `spell_training_comparisons.json`. Raw probability arrays/checkpoints/per-frame loss weights tetap ignored di `out/`. Tidak ada targetCSV, public/private evaluation baru, commit atau push otomatis. Sumber komponen dan perintah reproduksi: [README.md](README.md).
