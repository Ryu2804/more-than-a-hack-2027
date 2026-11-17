# Wave6: hysteresis membaik, belum lolos aturan promosi

Audit mencakup enam remote branch. Enam jalur menghasilkan36 alternatif +1 kontrol,851 evaluasi development pada23 scenarios/21 unique matches. Tidak ada training model baru. Seluruh hasil berasal dari decoding/fusion probabilitas frozen. [Audit branch dan daftar perbaikan](BRANCH_REVIEW.md), [rencana sebelum evaluasi](PLAN.md), [cara menjalankan](README.md).

## Keputusan frozen

Best alternative adalah **`m05_dead_hysteresis/low35_high50`**. Mean spellF1 development **0.674807**, versus wave5 **0.670099**, gain **0.004707**. Aturan sebelum run mensyaratkan gain minimal0.005, sehingga **tidak dipromosikan**. Selisih terhadap batas sekitar0.000293; batas tidak diturunkan setelah melihat hasil. Selected tetap wave5 `short_gru25/dead45`.

Paired match-bootstrap95% untuk gain best alternative adalah[0.003412,0.006096],5000draws,seed20261114. Ini interval development eksploratori yang tidak dikoreksi36-way search atau adaptasi lintas campaign. Lower positif tidak menggugurkan syarat minimum effect0.005. Tidak ada klaim statistical confirmation pada unseen matches.

| Provider development | Wave5 control | Best alternative | Selisih spellF1 |
|---|---:|---:|---:|
| IDSSE5 | 0.612109 | 0.609661 | -0.002448 |
| SkillCorner14 | 0.709337 | 0.718859 | +0.009522 |
| Metrica1/2 | 0.688852 | 0.695901 | +0.007049 |
| Equal mean provider | 0.670099 | 0.674807 | +0.004707 |

Dua provider menang, IDSSE turun sedikit. Worst development absolute count error best alternative42.66%, dibanding wave5 sekitar41.88%; masalah count belum selesai. MacroF1 development best alternative0.885634, control0.885922. Pilihan tidak dibalik berdasarkan macroF1 publik.

## Perubahan best alternative

Model dan joint probabilities tetap wave5:75% short-loss PP-TCN +25% frozen GRU. Tidak melatih ulang, mengubah player features, memakai label target, menambah touch detector atau recovered-ball predictions.

Satu perubahan: dead gate. Di setiap babak, ambil connected run dengan p_dead>0.35, minimal5samples pada5fps, lalu terima seluruh run hanya bila ada sample p_dead>0.50. Weak run tanpa strong seed ditolak. Sample sebelum strong seed dapat ikut dead, sehingga metode tetap offline. Original wave5 hanya menerima contiguous p_dead>0.45 dengan minimal1detik.

H/A Viterbi tetap reset per live interval, distance gate0.35m,kappa0.05,tau3, merge1s. Semua18lambda diperiksa dengan target rate dari training fold; count dihitung setelah merge. Tidak memakai dead-share shift atau aturan kickoff khusus target.

## Hasil enam jalur

| Jalur | Konfigurasi terbaik development | SpellF1 | Selisih vs control |
|---|---|---:|---:|
| Joint blends | boundary10 | 0.669565 | -0.000534 |
| Visibility-adaptive blend | visible100_missing65 | 0.669300 | -0.000799 |
| Dead-head-only blend | boundary75 | 0.670813 | +0.000713 |
| Occlusion switch costs | factor1.25_window1 | 0.668912 | -0.001188 |
| Dead hysteresis | low35_high50 | 0.674807 | +0.004707 |
| Low-confidence short guard | max1_confidence60 | 0.670099 | 0.000000 |

Tidak ada konfigurasi eligible. Enam hysteresis settings menempati enam peringkat teratas tetapi seluruh gain masih<0.005. Blending lebih konservatif, visibility gating dan switch penalty lebih besar tidak otomatis membantu. Short guard terbaik menghasilkan skor sama dengan kontrol, bukan peningkatan.

Seluruh37 configuration records dan851per-scenario scores: [wave6_development.json](../results/wave6_development.json).36 konfigurasi bukan36 arsitektur atau training runs. Development runtime82.344detik di script, CPU satu proses, training_seed_fits0: [execution](../results/wave6_execution.json).

## Holdout setelah freeze

Tiga role fixed: selected,best_new,previous_wave5. Selected dan previous_wave5 identik, sehingga hanya dua pipeline berbeda.48 role/scenario rows mencakup16scenarios dari9unique matches; stress6 berasal dari6SkillCorner yang sama. Semua match dan transformasi stress sudah terekspos pada campaign lama.

| Cohort | Wave5 control | Best hysteresis | Selisih mean spellF1 |
|---|---:|---:|---:|
| IDSSE2 | 0.667175 | 0.663522 | -0.003652 |
| SkillCorner6 clean | 0.678304 | 0.684029 | +0.005726 |
| SkillCorner6 stress | 0.663030 | 0.667709 | +0.004679 |
| Metrica3 dua kondisi | 0.690016 | 0.698456 | +0.008440 |

Best alternative menang7/10ordinary scenarios, atau6/9unique match means. Mean gain scenarios biasa0.004393. Pada paired ball-drop stress menang6/6, mean gain0.004679. Ini memperbaiki stress dibanding wave5, tetapi tidak menjadi alasan memilih ulang kandidat yang gagal syarat development. Metrica3 clean/r0 memakai GT match sama, bukan dua pertandingan independen.

### Precision dan long possessions pada stress

Diagnostic one-to-one official matching pada IoU0.75, dipool dari enam paired stress:

| Diagnostic | Wave5 | Best hysteresis |
|---|---:|---:|
| Predicted possessions | 1713 | 1669 |
| Matched possessions | 751 | 760 |
| Precision | 43.84% | 45.54% |
| Recall GT<4detik | 18.69% | 20.00% |
| Recall GT>=15detik | 68.64% | 68.12% |

Predicted count turun44 dan matched count naik9; precision serta short recall membaik. Long-spell recall tetap sedikit turun, sehingga tidak ada klaim seluruh robustness tradeoff selesai. GT seluruh1689spells tetap masuk scorer utama. Recall IoU0.75 ini diagnostic, bukan pengganti headline mean IoU0.3..0.7.

[Holdout lengkap](../results/wave6_holdout.json), [comparison aggregates](../results/wave6_comparison.json), [duration diagnostics](../results/wave6_duration_diagnostics.json).

## Public diagnostic: skor baru, bukan kandidat yang dipromosikan

Public GT hanya7500frames/17possessions. Hanya tiga role fixed dievaluasi setelah selection, holdout dan selected rebuild; selected/previous identik. Tidak ada36-way public sweep.

| Pipeline berbeda | SpellF1 publik | MacroF1 publik | Predicted possessions |
|---|---:|---:|---:|
| Selected wave5 | 0.772414 | 0.902059 | 12 |
| Best hysteresis, exploratory | **0.786207** | **0.925369** | 12 |

Best alternative memberi diagnostic spellF1 publik tertinggi yang tercatat pada campaign kita, naik0.013793. **Tetap tidak dipromosikan**: minimum effect development gagal sebelum public dibuka. Public gain tidak membuktikan private improvement dan tidak dipakai untuk menurunkan adoption threshold. Skor private/full-match belum diketahui.

Best alternative CSV lokal: `wave6/out/candidates/best_new.csv`.

SHA-256: `2bf7cc52306740e4af5ca1e34c07d843e3d7705b0f166ad051d414bd5f4c5acf`.

Selected tetap `wave6/out/candidates/selected.csv`, identik kandidat wave5.

SHA-256: `cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8`.

[Audit publik](../results/wave6_public.json).

## Verifikasi yang benar-benar dijalankan

- 40 regression cases: 39 lulus pada batch awal; satu gagal karena `import run` meresolve runner wave5, bukan wave6. Test import diganti unique module identity, lalu hanya kasus yang gagal dijalankan ulang dan lulus. Assertion tetap sama. Seluruh perbaikan selesai sebelum preregistration dan evaluasi nyata.
- Checks mencakup joint-vs-conditional mixtures, zero-live heads, half-local visibility, actual variable-cost decoder parity, hysteresis strict thresholds/min duration/half reset, simultaneous A-B-A guard dan strong-short preservation, count calibration setelah custom postprocess, serta bootstrap clustering conditions dalam match.
- Actual CLI dev menjalankan37settings pada23scenarios dan mereproduksi control objective0.670099305497763 persis. Holdout menjalankan48role/scenario evaluations; official public scorer menerima kedua CSV berbeda.
- Proses terpisah memuat ulang cached probabilities dan mengulang selected decoding. Target199319baris identik SHA dengan wave5. Ini selected **decoding rebuild, bukan model refit**. Best alternative belum memiliki independent model refit baru; tidak ada fit baru untuk pipeline mana pun.
- Frozen source/input/selection hashes diperiksa pada tahap holdout, rebuild dan public. Selection SHA `e387c08beef727217ffeb8385eeca722c1348a554d38e2b087cd1b4c2fa099ec`.

[Bukti rebuild](../results/wave6_verification.json) dan [catatan regression/final checks](../results/wave6_checks.json). Temporary rebuild CSV dihapus setelah hash comparison. Tidak ada source/PLAN/input edits setelah preregistration.

## Perbaikan repo yang diterapkan dan yang tidak

Wave6 menambah cache fail-closed sebelum inherited inference, exact conditional-head preservation saat live mass0, bootstrap per unique match, common decoder API untuk enam jalur, hash checks dan explicit distinction selected/exploratory/public/private. Tidak ada fallback yang menulis hasil campaign lama.

Audit menemukan registry consolidation keliru memberi label beberapa **dead-share** Jeremy sebagai public spellF1; benchmark Farras di sana masih adapter wave2; Rasya mempunyai speed-unit bug dan held-out GT orientation; Ryu mempunyai nearest-opponent feature defect yang juga diwarisi snapshot kita. Detail, source dan risiko ada pada [BRANCH_REVIEW.md](BRANCH_REVIEW.md). Temuan tersebut **tidak** diklaim sudah diperbaiki pada branch teammate. Memperbaiki learned feature membutuhkan regeneration+retraining, bukan inference-only patch pada weights lama.

Semua file baru dipisah di wave6 dan aggregate results; indeks Farras diperbarui. Source teman dan wave1..5 tetap utuh. Raw tracking, probabilities/labels per frame, weights serta submission CSV tetap ignored. Tidak ada commit/push otomatis.
