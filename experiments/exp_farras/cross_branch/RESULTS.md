# Cross-branch: primary lolos, ensemble lebih sederhana mencetak mean tertinggi

Semua eksperimen selesai: tiga model contextual baru,7LOSO folds,63fresh neural seed fits/31500updates,28probability/decoder settings,196per-match evaluations dan7native-rule diagnostics. Tidak ada fit SetSeq CPU baru, old weights, GT publik atau target submission. [PLAN](PLAN.md), [adaptasi source tiap branch](SOURCES.md), [cara menjalankan](README.md).

## Keputusan yang dibekukan sebelum run

Primary `triple_combined/hysteresis` lolos aturan promosi **untuk riset**:

| Pipeline | Mean spellF1 IDSSE | Mean macroF1 |
|---|---:|---:|
| Control short-loss PP-TCN/dead45 | 0.595755 | 0.902935 |
| Primary gabungan semua komponen | **0.609030** | **0.909290** |
| Best raw alternative: short+SetSeq25/dead45 | **0.615458** | **0.909587** |

Primary memakai50%short-loss PP-TCN,25%SetSeq,25%contextual TCN gabungan dengan fitur kontak Jeremy +touch-memory/dead-cue Rasya. Backbone contextual TCN memakai PP encoder Ryu,3seeds dan budget500updates yang sama dengan benchmark sebelumnya. Hysteresis low.35/high.50 tetap seperti PLAN.

Gain primary terhadap control **0.013274**, paired-bootstrap95% **[0.005245,0.021057]**, menang **6/7matches**. Tiga syarat sebelum run terpenuhi:gain>=.005,>=5wins,lower>0. Ini bukan bukti private improvement; match dan hypotheses historis sudah terekspos.

Best raw alternative yang lebih sederhana memakai **75%short-loss PP-TCN+25%SetSeq**, decoder dead45. Mean0.615458, gain0.019702, interval[0.015308,0.024763], menang7/7. Setting ini sudah ada pada grid sebelum hasil diketahui, tetapi tidak menggantikan primary secara post-hoc. Disimpan sebagai alternatif eksploratori yang layak validasi lanjutan. Tidak mengklaim best dari28settings sebagai unbiased heldout score.

**Menggabungkan semua fitur tidak otomatis terbaik.** Komponen baru membantu beberapa kombinasi, tetapi dua model cached yang saling melengkapi memberi hasil tertinggi. Tidak menyembunyikan fakta ini demi membenarkan training tambahan.

## Adaptasi yang benar-benar dilakukan

- Ryu: actual PP-TCN implementation, player-set attention, view symmetry dan conditional dead/team probabilities. Shipped27-match weights tidak dipakai karena mengandung semua IDSSE benchmark matches.
- Jeremy: true25fps contact globals dari adopted touch-only lineage exp3/4/5; ditambah probabilitas SetSeq OOF benchmark sebelumnya. Ini bukan exact exp5 model atau exp7 recovery. Contact globals dipelajari dalam TCN GPU, bukan mengulang GRU CPU.
- Rasya: last-touch carriedteam, recencyexp(-age/10s), sustained out/slow cue dengan short-dead suppression6s. Semua dihitung pada raw25fps observasi dengan metadata tim, tanpa GT orientation/dead/X mask. Speed dari posisi/elapsedseconds, tidak dibagi3.6.
- Farras: short-spell-loss model OOF, fixed joint ensembles, dead45 versus hysteresis yang telah dipreregistrasi. GT provider tetap asli, termasuk semua short spells.
- Main/consolidation: official scorer dan shared-evaluation contract. Tidak memakai ranking adapter lama atau angka dead-share yang salah diberi label public spellF1.

Raw25fps di-regenerate dari cached clean observations dengan seed corruption canonicalr0. Labels dihapus sebelum regeneration; sampled positions/ball/team/timeline diperiksa sama dengan prepared5fps. Ketujuh match lolos. Contact/memory menambah informasi sub-step di antara5fpssamples: ini feature-information augmentation, bukan klaim tensor input identik dengan baseline lama.

## Ablations lengkap

Semua angka berikut mean official spellF1 atas7heldoutmatches. Tidak menghilangkan GT short spells atau mengganti label definition.

| Probability family | Dead45 | Hysteresis |
|---|---:|---:|
| short control | 0.595755 | 0.595994 |
| SetSeq | 0.578602 | 0.580111 |
| PP-TCN | 0.571090 | 0.570414 |
| contact-only TCN | 0.579219 | 0.579810 |
| memory-only TCN | 0.581164 | 0.583277 |
| contact+memory TCN | 0.586845 | 0.584343 |
| short75+SetSeq25 | 0.615458 | 0.614183 |
| short75+PP25 | 0.593436 | 0.594299 |
| short75+contact25 | 0.596079 | 0.597846 |
| short75+memory25 | 0.596968 | 0.597868 |
| short75+combined25 | 0.597035 | 0.597924 |
| short50+SetSeq25+contact25 | 0.605333 | 0.604767 |
| short50+SetSeq25+combined25 | 0.610665 | 0.609030 |
| visibility-weighted combined+short | 0.598298 | 0.599283 |

Tiga fresh contextual arms punya arsitektur36channels, PP encoder, seeds,500updates dan sampling yang sama. Contact-only mengosongkan3memory channels, memory-only mengosongkan4contact channels, combined memakai7channels. Dibanding cached baseline29channel, ada perbedaan informasi/dimensi/loss: jangan mengatribusikan seluruh selisih sebagai efek satu fitur saja.

Temuan negatif dipertahankan:

1. Ketiga contextual TCN standalone belum mengalahkan short-loss control secara mean.
2. Menambah PP25 ke short tidak membantu; ensemble bukan jaminan.
3. Quality/visibility blend tidak mencapai gain minimum primary.
4. Hysteresis hampir datar pada short control dan justru menurunkan beberapa kombinasi. Diagnostic public wave6 yang tinggi tidak menjamin decoder itu unggul di IDSSE.
5. Primary dengan dead45 mencapai0.610665, sedikit di atas primary hysteresis0.609030; decoder primary tidak diganti setelah melihat angka ini.

## Native Rasya-style diagnostic

Rule dari raw25fps yang dikoreksi dan di-sample ke5fps mencapai mean spellF1 **0.322034**. Ini bukan hasil original Rasya0.5263/0.5057 dan bukan shared probabilistic decoder. Raw source/temporal resolution, suppression dead6s, label-free orientation dan missingness berbeda dari adapter fair sebelumnya. Tidak ada ablation yang mengisolasi masing-masing perubahan rule. Neural models mendapat cue/memory sebagai fitur yang bisa diabaikan, bukan hard override dari rule yang masih lemah.

## Pemisahan data dan pemilihan

Semua learned fresh models setiap fold melatih persis6match lain. Original short,SetSeq,PP probabilities dipakai hanya pada heldout identity dan originalmanifest yang cocok; tidak ada final model inferred pada training games. Normalisasi dan decoder rate dari training fold saja. Tidak ada learned meta model, label-based provider gate atau bobot yang dioptimalkan dari outer labels.

Primary ditentukan sebelum full fits; semua28settings tetap dilaporkan. Bootstrap5000draws seed20261115 dengan unit unique match, bukan frame/seed. Interval unadjusted dan penelitian adaptif di tujuh match yang sama; bukan nested tuning evaluation atau fresh external confirmation. Tidak mengklaim unggul atas exact submission semua anggota tim.

## Verifikasi dan runtime

- 31 semantic tests lulus: sub-step ball impulse, earliest tie, missingness/half isolation, team symmetry, touch duration/recency, correct metric speed, dead suppression149/150frames, strict regeneration identity, joint fusion, post-merge count calibration dan full18lambda grid.
- First prepare gagal karena duplicate `seed` metadata key. Komposisi audit diperbaiki sebelum registry full fits; prepare seluruh7match dan actual GPU smoke ketiga model kemudian selesai exit0. Tidak ada source/PLAN/input edits setelah registry.
- Smoke9seed fits memakai1step, dipisahkan dari63full seed fits dan tidak dipakai sebagai performance estimate.
- Full fits dilakukan bertahap, setiap family/fold pada child process tersendiri. Seluruh21training jobs selesai exit0. Actual metadata menunjukkan6unique train names, test excluded,3seeds/500steps; tidak sekadar menulis intended split.
- Aggregate waktu di dalam training jobs1424.704detik, tidak mencakup semua startup/orchestration overhead. Tidak melatih SetSeq CPU ulang; hanya memakai OOF cache yang diverifikasi.
- Original control mean **0.5957554001098855** direproduksi persis.
- Independent process rebuild memverifikasi **203decoded arrays identik**:196model-setting/match outputs dan7native diagnostics. Ini decoding rebuild, bukan refit.

Manifest SHA: `dce422c8eb38801630c4909f648274e0b793ceadf5de001e614ab57b9c71b572`.

Bukti: [seluruh196scores](../results/cross_branch_results.json), [protocol](../results/cross_branch_protocol.json), [source regeneration](../results/cross_branch_preparation.json), [actual training membership](../results/cross_branch_training.json), [checks](../results/cross_branch_checks.json), [independent rebuild](../results/cross_branch_verification.json).

## Status output

Primary lolos untuk riset, best simpler ensemble tetap alternatif eksploratori. Tidak ada public evaluation baru, CSV submission target baru, claim private score, commit atau push. Semua code baru terpisah di cross_branch; old campaigns dan source teammate tetap utuh. Raw tracking, labels/probabilities per frame, controls, native arrays dan weights tetap ignored. Hanya source/docs dan6aggregate JSON diekspor.
