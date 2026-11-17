# Benchmark IDSSE dengan split bersama

[PLAN.md](PLAN.md) menetapkan data, budget dan scoring sebelum full fits. Lima keluarga: HGB symmetric, SetSeq, PP-TCN, short-loss PP-TCN, dan label-free touch-hold. Ini adapted-family benchmark, bukan reproduksi exact submission anggota tim.

Full benchmark selesai: mean spellF1 short-loss PP-TCN **0.595755**, SetSeq0.578602, PP-TCN0.571090, HGB0.315719. Short-loss menang mean tetapi interval perbandingan dengan SetSeq masih melintasi0.32semantic tests lulus;63neural seed fits baru dan7HGBfits selesai,63decoded arrays direproduksi identik dalam proses terpisah. [Hasil lengkap dan keterbatasan](RESULTS.md).

Semua memakai tujuh prepared IDSSE r0 matches byte-for-byte,7leave-one-match-out folds. Setiap learned arm dilatih ulang dari nol pada6match lain. Tidak ada pretrained checkpoints, public GT, target prediction atau pemilihan hyperparameter dari outer fold. Provider labels asli, termasuk short possessions, dipakai scorer resmi. Match sudah terekspos penelitian sebelumnya; bukan fresh blind validation.

Empat learned arms memakai decoder sama (wave5 dead45), sekaligus argmax diagnostic yang sama. Touch-hold masuk panel native-rule karena mengeluarkan hard labels. Jangan mengklaim rule menerima decoder probabilistik identik. Neural seeds0/1/2,500updates/seed,batch16,window384; equal-update budget bukan equal FLOPs atau recipe produksi1500steps. Optimizer/augmentation/representasi berbeda dan tercatat pada PLAN.

## Menjalankan

Dari root repo, Python3.10 dengan dependencies wave5; perlu prepared IDSSE packs wave3 dan feature packs wave4 untuk ketujuh match. Tidak mengunduh ulang atau mengubah cache lama.

```powershell
python -B -m unittest discover -s experiments/exp_farras/fair_idsse -p 'test_*.py' -v
python -B experiments/exp_farras/fair_idsse/check.py
python -B experiments/exp_farras/fair_idsse/run.py all
python -B experiments/exp_farras/fair_idsse/run.py rebuild
```

`check.py` benar-benar fit/predict data nyata dengan1update/seed pada tiga neural arms, ditambah HGB/rule. Hasil smoke bukan estimasi performa. `all` menjalankan63neural seed fits/31500updates dan7HGB fits; seluruh63score rows wajib selesai. Cache baru menyertakan identity fold, training names, config dan manifest SHA; mismatch ditolak. `all` menolak menimpa results yang sudah lengkap, tetapi dapat melanjutkan job identik yang selesai jika proses sebelumnya terputus. Jangan mengubah source/PLAN/hash untuk menerima cache tak cocok.

Tiap family/fold dijalankan pada child process terpisah. SetSeq memakai CPU karena child CUDA tetap keluar fatal setelah menghitung; dua PP-TCN memakai CUDA, HGB/rule CPU. Parent hanya scoring dan menerima cache sesudah child exit0. Backend per family diikat registry. Budget updates/seeds/data tetap sama; jangan memakai hasil ini untuk ranking latency atau equal-FLOP efficiency.

`rebuild` proses terpisah memuat output probabilitas dan mengulang semua decoder, membandingkan63hash array label. Ini decoding rebuild, bukan refit. Tidak menghasilkan submission publik dan tidak otomatis promote kandidat.

## Output dan sumber

`out/` ignored menyimpan preregistered.json, smoke gabungan/isolated, job probabilities/metadata/checkpoints, results.json dan verification.json. Smoke gabungan yang keluar fatal disimpan sebagai bukti kegagalan, bukan pass. Hanya metrik per-match, mean, paired bootstrap dan source/input digests aman diekspor ke `../results/fair_idsse_*.json`. Jangan membagikan arrays tracking/labels/probabilities atau weights. Tidak ada commit/push otomatis.

IDSSE CC BY4.0, metadata team Home/Away dan corruption provenance pada wave3/cache/idsse_audit.json. Credit kode/arsitektur/data: [wave3 source](../wave3/reference/SOURCE.md), [wave4 source](../wave4/reference/SOURCE.md). Kedua PP arms memakai builder aktual `wave4/features.py`, yang sudah menghitung opponent relatif tim pemain. Klaim audit sebelumnya bahwa bug file referensi Ryu otomatis diwarisi model kita dikoreksi pada PLAN; file referensi dan builder model bukan fungsi yang sama.
