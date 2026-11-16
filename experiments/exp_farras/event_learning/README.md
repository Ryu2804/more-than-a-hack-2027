# Event learning

Enam training recipes di atas event sampler: balanced team loss, focal team loss, auxiliary boundary supervision, signed change-point ranking, EMA weights dan observed contact inputs. [PLAN.md](PLAN.md) membekukan hipotesis, konfigurasi, strong research control dan aturan promosi sebelum full training. Folder lama dan eksperimen anggota lain tidak diubah.

## Menjalankan

Dari root repo, Python3.10 dengan environment fair_idsse/spell_training, PyTorch, Numba dan threadpoolctl. Perlu canonical r0 packs/PPfeatures, audited short/SetSeq OOF probabilities, cached previous event models/probabilities/replay proofs dan raw25fps control cache cross_branch. Tidak mengunduh data atau memuat predecessor weights untuk training baru.

```powershell
python -B experiments/exp_farras/event_learning/check.py
python -B experiments/exp_farras/event_learning/run.py register
python -B experiments/exp_farras/event_learning/run.py fit-fold --match J03WMX --group first
python -B experiments/exp_farras/event_learning/run.py fit-fold --match J03WMX --group second
python -B experiments/exp_farras/event_learning/run.py all
python -B experiments/exp_farras/event_learning/run.py rebuild
```

`check.py` menjalankan semantic tests lalu18actual smoke seedfits,1update per seed, enam keluarga. Setiap smoke ensemble memprediksi full heldout match; proses lain memuat checkpoint dan menuntut probabilities identik. Smoke tidak dipakai sebagai performance estimate.

`register` membekukan source/PLAN/input/provenance hashes. Full recipe: tujuh folds, enam training matches/fold, tiga seeds,500updates/seed,batch16,crop384. Total126fresh seedfits/63.000updates. `first` menjalankan balanced/focal/boundary; `second` ranking/ema/contact; pemisahan hanya orchestration, bukan protokol atau urutan penilaian. Semua GPU jobs berurutan pada child processes terpisah. Parent boleh mengembangkan slice secara paralel tanpa menjalankan banyak training GPU bersamaan.

Setiap fit menyimpan actual fold/seed/update/parameter audits, model weights dan prediction digests. Proses inference replay memuat setiap ensemble tersimpan dan mengecek exact equality terhadap full heldout probability arrays. `all` memverifikasi/reuse hanya jobs dengan identitas/hash tepat, menyelesaikan semua folds lalu evaluate. Completed results tidak ditimpa. `evaluate` boleh dipanggil terpisah setelah semua42jobs dan replay proofs lengkap.

`rebuild` pada proses lain membangun ulang280decoded arrays dan memverifikasi42model-replay proofs, termasuk byte-identical controls. Ini reproduksi inference/decode, bukan full training refit kedua. Exit nonzero tetap failure walaupun proses menulis sebagian artefak.

## Kontrol dan keputusan

Strong research control memakai short50% + SetSeq25% + previous event25% dengan frozen consensus(alpha0.5/contact_beta0.5), mean official spellF1 IDSSE0.619562844991005. Ini winner exploratory sebelumnya, bukan deployment promotion retroaktif. Legacy short75% + SetSeq25%/dead45 tetap dilaporkan terpisah0.6154575861548992.

Primary baru `balanced_replace/consensus`: ganti25%previous event dengan25%event+balanced-loss model. Promosi hanya jika gain>=0.005 terhadap strong control, menang>=5/7matches dan paired match-bootstrap95%lower>0. Tidak memilih otomatis raw winner saat primary gagal. Semua40settings ditentukan sebelum run; tidak retune fusion/decoder dari hasil baru.

Empat recipes memakai135.219parameter dasar. Boundary menambah auxiliary output; contact menambah tujuh input channels; EMA menyimpan training shadow. Equal update counts tidak berarti parameter/input/compute parity penuh. Boundary target dihitung per full training half sebelum crop; original short-loss dan complete-spell weighting mempunyai aturan censoring berbeda yang sengaja dipertahankan. Boundary auxiliary head tidak dipakai decoder. EMA dipilih hanya berdasarkan recipe tetap, bukan heldout scores.

Semua pertandingan dan hipotesis sudah terekspos; bukan fresh blind estimate. Interval bootstrap nominal tidak mengoreksi multiple comparisons atau adaptive history. Hasil IDSSE tidak dibandingkan langsung ke public/private score. Tidak ada target CSV, evaluasi public/private baru, commit atau push otomatis.

## Artefak dan sumber

Hasil lengkap: [RESULTS.md](RESULTS.md). Aggregate exports di `../results/event_learning_*.json`. `out/` menyimpan checkpoints baru, raw predictions, audits dan replay proofs; semuanya ignored. Raw tracking, labels, weights dan per-frame outputs tidak dibagikan.

Sumber model, feature dan benchmark: [spell_training](../spell_training/README.md), [fair_idsse](../fair_idsse/README.md), [cross_branch](../cross_branch/SOURCES.md), [temporal_methods](../temporal_methods/README.md). Feature builder aktual memakai opponent mask per player yang benar; tidak menyalin bug pada file referensi Ryu lama.
