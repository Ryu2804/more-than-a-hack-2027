# Audit paparan publik dan batas generalisasi

## Yang terverifikasi dari kode lama

Competition RULES.md52-84 melarang training/tuning pada human labels dari provided match; evaluation public window hanya untuk pipeline correctness. EVALUATION.md125-142 menjelaskan17possessions tidak cukup untuk memilih model dari ranking publik.

Inspected selection logic memakai external-development metrics dan membekukan pilihan sebelum public diagnostic scoring:

| Campaign | Selection eksternal | Akses public setelah freeze |
|---|---|---|
| wave3 | run.py241-273 | run.py417-438, GTload423 |
| wave4 | run.py172-210 | run.py363-383, GTload370 |
| wave5 | run.py224-240 | run.py387-409, GTload396 |
| wave6 | run.py125-139 | run.py212-233, GTload222 |

Tidak ada aliran public metrics kembali ke selection branches tersebut. Wave6/RESULTS.md7 mempertahankan wave5 saat candidate gagal external minimum gain; public diagnostic0.786207 tidak membalik keputusan. Ini bukti urutan keputusan dalam kode yang diinspeksi, **bukan jaminan skor publik yang sudah diketahui tidak pernah memengaruhi hipotesis berikutnya**. Wave5/PLAN.md5-7 dan wave6/PLAN.md30-32 sendiri menyebut paparan dan riset adaptif.

Recent fair_idsse/cross_branch/spell_training/event_learning runners menilai eksternal dan mencatat `new_public_evaluation=False`. Flag itu berlaku untuk campaign tersebut, tidak menghapus public access historis. Menggunakan best exploratory sebelumnya sebagai control baru juga bentuk adaptive research yang sudah dilaporkan.

## Semua27match eksternal sudah terekspos

Wave4/campaign.py15-24 mendefinisikan7IDSSE dan20SkillCorner. Wave3/PLAN.md15 mengakui tujuh IDSSE pernah dipakai Jeremy. Wave4/PLAN.md9-15 mencatat shipped weights melatih27matches; historical SkillCorner holdouts juga pernah dipakai Ryu. Wave5/RESULTS.md60-64 mengakui paparan holdout itu. Mengganti nama split tidak menjadikannya blind.

[Current SkillCorner match tree](https://api.github.com/repos/SkillCorner/opendata/git/trees/0c809ab9eabe28ebcd0b28a6916617cde88f852c) berisi20IDs yang sama dengan SC_IDS lokal. Tidak ada fresh cohort dari inventory ini. Dokumentasi README upstream menyebut10matches, tetapi tree dan matches.json memuat20; cohort ditentukan dari inventory aktual, bukan angka README. Tidak mengunduh tracking baru.

## Direct leakage versus label-access barrier

Feature path yang diinspeksi memisahkan possession labels dari observasi:

- wave3/reference/jeremy_data.py125-162 memisahkan metadata/team mapping, tracking dan labels; corruption198-276 menerima observasi, bukan labels.
- wave3/prepare_idsse.py293-317 membangun X16 sebelum ekstraksi labels.
- wave4/prepare_skillcorner.py457-497 memisahkan phase labels dari interpolation/distances/X16.
- wave1/data.py76-113 membangun X16 dari posisi, jarak, visibility dan motion/gaps.
- wave4/features.py530-576 membangun PP/pooled/ball features dari observasi; direction inference121-161 memakai posisi pemain, bukan kecocokan labels.
- Historical target carrier wave3/prepare_local.py339-381,394-418 memakai labelsNone kemudianunknown-1.

Tidak ditemukan penggunaan possession GTsebagai feature pada path tersebut. Namun inherited loader wave4/campaign.py63-80 **membaca y/y25sebelum observations() membuangnya**. Itu tidak membuktikan labels memengaruhi inference, tetapi tidak memenuhi klaim literal bahwa inference tidak dapat membaca test labels. New campaign memakai carrier tanpa y/y25dan runtime read guard terhadap original label-bearing packs/vault.

Inherited training helper wave4/campaign.py76-78 juga menambahkan r0/r1/r2untuk IDSSE. New source-only protocol memanggil loader r0secara eksplisit; helper itu tidak digunakan.

## Yang boleh dan tidak boleh disimpulkan setelah eksperimen baru

Boleh, jika semua verification berhasil: training fresh source-provider-only; match/provider disjoint per arah; normalization/rate/weights source-only; inference label-free; semua predictions committed sebelum metric baru; competition-public read ditolak oleh trusted-code hook; hasil pada external datasets yang sudah terekspos.

Tidak boleh: bebas overfit, tidak pernah melihat public, fresh blind test, unbiased performance, peningkatan private/full-match, atau OS-level confidentiality guarantee. Python audit hooks tidak mencakup semua native I/O, subprocess leakage atau adversarial bypass. New clean/gap conditions adalah paired variants pada27matches, bukan54independentmatches. Provider transfer mencampur domain shift, definisi label/proxy dan kualitas tracking; skor antar provider bukan bukti overfit sendirian.

Keputusan utama dan metode baru: [PLAN.md](PLAN.md). Hasil eksekusi dan verifikasi: [RESULTS.md](RESULTS.md). Data kompetisi publik tidak dibuka oleh audit ini; bukti aturan diambil dari dokumentasi dan kode, bukan isi GTCSV.
