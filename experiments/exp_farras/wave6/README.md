# Wave6: conservative fusion dan occlusion-aware decoder

[Audit seluruh branch](BRANCH_REVIEW.md) dan [rencana sebelum evaluasi](PLAN.md). Enam jalur,36 alternatif ditambah exact wave5 control. Tidak ada training seed fit baru: probabilitas OOF/final dari wave5 dipakai frozen untuk mengisolasi perubahan fusion/decoder. Scorer resmi dan seluruh short GT tetap sama.

Hasil: best hysteresis spellF1 development **0.674807**, diagnostic publik **0.786207**, macroF1 publik0.925369. Gain development0.004707 gagal batas promosi0.005; selected tetap wave5. Best alternative menang6/6 paired stress, tetapi IDSSE sedikit turun dan long-spell recall belum pulih.40 regression cases lulus setelah satu test-import fix; selected decoding rebuild199319baris identik. [Laporan lengkap](RESULTS.md). Tidak ada klaim peningkatan private.

## Menjalankan

Dari root repo, Python3.10 dan dependency campaign sebelumnya. Wajib tersedia cache/data/features, manifest dan seleksi frozen wave3..5, termasuk cached short-TCN, boundary-TCN, oldPP dan GRU pada seluruh development/holdout/stress/target. Tidak ada fallback yang meregenerasi missing predictions atau menulis campaign lama. Target tetap199319frame.

```powershell
python -m pip install -r experiments/exp_farras/wave6/requirements.txt
python -B -m unittest discover -s experiments/exp_farras/wave6 -p 'test_*.py' -v
python -B experiments/exp_farras/wave6/run.py dev
python -B experiments/exp_farras/wave6/run.py holdout
python -B experiments/exp_farras/wave6/run.py rebuild
python -B experiments/exp_farras/wave6/run.py public
```

`dev` menolak overwrite frozen selection. Sebelum evaluasi, source/PLAN/input hashes didaftarkan. Tahap berikut memverifikasi hash, bukan menerima cache hanya karena file exists. Evaluasi CPU satu proses dengan threadpool1; tidak perlu mengulang CUDA training. Dependencies Torch tetap dibutuhkan oleh modul inherited. Backend training lama pada manifest tetap menjadi provenance, bukan klaim training wave6.

`rebuild` wajib proses terpisah setelah holdout. Ia memuat ulang probabilitas dan mengulang selected decoding, menulis CSV sementara untuk membandingkan SHA lalu menghapusnya. Ini independent decoding rebuild, **bukan learned-model refit**. Model refit evidence diwarisi wave5/wave3.

## Protokol

- Development23 scenarios/21 unique matches. Training-fold probabilities/rate tetap dari19 IDSSE+SkillCorner matches,6 grouped folds. Metrica1/2 zero-shot; tidak dilatih ulang.
- Objective mean spellF1 per match, per provider, lalu equal mean tiga provider. MacroF1 tie-breaker.
- Promosi minimal+0.005, minimal2/3 provider menang, tidak ada provider turun>0.01, lower paired-bootstrap95%>0. Bootstrap5000,seed20261114, unique match di dalam provider. Interval eksploratori, belum dikoreksi multiple search atau adaptasi campaign lama.
- Holdout lama9 unique matches/10ordinary scenarios, ditambah6paired SkillCorner stress dari match sama. Seluruhnya sebelumnya terekspos. Bukan fresh blind validation.
- Hanya selected,best_new,previous_wave5 diuji setelah freeze; role identik tidak dihitung sebagai model independen. Public17GT possessions hanya audit format/alignment, bukan dasar pemilihan. Private score belum diketahui.

## Artefak

`out/` ignored: preregistered.json, selection.json, execution.json, holdout_report.json, duration_diagnostics.json, verification.json, public_audit.json dan candidates/*.csv. Source predictions tidak disalin ke campaign baru.

Hanya kode, dokumen dan metrik agregat aman dibagikan. Raw tracking/labels/probabilitas per frame, weights dan submission CSV tidak dipush. Tidak ada commit/push otomatis. Credit/source licenses diwarisi [wave4/reference/SOURCE.md](../wave4/reference/SOURCE.md): SkillCorner MIT dan IDSSE CC BY4.0, Metrica hanya evaluation proxy pada model terpilih.
