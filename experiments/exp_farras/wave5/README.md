# Wave5: boundary, short-possession dan masking

[PLAN.md](PLAN.md) menetapkan96 konfigurasi dan aturan adopsi sebelum training. Ketiga model baru memakai PP-TCN wave4 dengan intervensi terpisah: auxiliary boundary head, double loss weight untuk short live spells, atau full ball masking lebih sering. Decoder membandingkan kalibrasi sebelum/sesudah merge dan reset H/A pada dead intervals. Campuran quality memakai visibility/entropy, bukan label target.

Hasil: `short_gru25/dead45` lolos aturan adopsi dengan spellF1 eksternal0.670099. SpellF1 publik tetap0.772414; macroF1 kandidat0.902059. Short-spell recall membaik, tetapi mean spellF1 paired ball-drop stress sedikit turun.10 regression tests dan actual CUDA fitting/checkpoint/refit lolos. Hasil negatif, seluruh96 konfigurasi, dan hash reproduksi: [RESULTS.md](RESULTS.md).

## Status metode utama

Wave5 `short_gru25/dead45` ditetapkan sebagai metode utama branch `farras`. Gunakan [submission/README.md](../../../submission/README.md) untuk ekspor CSV yang dikunci hash dan status kesiapan. Source, PLAN, selection, model, dan output frozen campaign tidak diubah oleh penetapan ini; bukan training atau public evaluation baru.

Ekspor cukup dengan `python -B submission/export.py` dari root repo; CSV prediksi terpilih disertakan di `submission/artifacts/farras_wave5.csv` untuk harness. Prosedur di bawah untuk reproduksi campaign dengan cache/model lama, bukan langkah wajib setiap ekspor dan belum merupakan jalur raw bundle end-to-end.

## Menjalankan

Dari root repo, Python3.10. Cache/data/features dan hasil frozen wave3/4 harus tersedia. Tidak mengubah artefak lama atau mengunduh dataset baru. SkillCorner MIT, IDSSE CC BY4.0; kredit source/data pada [wave4/reference/SOURCE.md](../wave4/reference/SOURCE.md). Metrica hanya evaluasi, tanpa anchor tree pada family wave5.

```powershell
$env:FARRAS_SOURCE_EXPERIMENTS = "D:\workspace\sumber\experiments"
$env:FARRAS_TRAIN_DEVICE = "cuda"
python -m pip install -r experiments/exp_farras/wave5/requirements.txt
python -B experiments/exp_farras/wave5/test_invariants.py
python -B experiments/exp_farras/wave5/run.py dev
python -B experiments/exp_farras/wave5/prepare_stress.py
python -B experiments/exp_farras/wave5/prepare_reference.py
python -B experiments/exp_farras/wave5/run.py holdout
```

`python -B experiments/exp_farras/wave5/run.py public` adalah audit diagnostik campaign setelah freeze, bukan bagian ekspor atau syarat menjalankan metode utama. Audit publik yang sudah tercatat tidak perlu diulang untuk penetapan metode.

`prepare_reference.py` memakai CPU dalam proses terpisah agar GRU lama tidak memicu shutdown CUDA lokal dan modul checkpoint `models` tidak bertabrakan dengan TCN. Jangan mengimpor model GRU dan TCN dengan nama modul sama dalam satu proses. CUDA TCN harus lolos actual fitting/inference smoke; backend dihash dalam registry.

## Protokol

- Training19 match dari wave4,6 grouped folds,3 seed/model,1500 steps/seed. Tiga model baru memberi21 ensemble fits/63 seed fits, bukan96 model independen.
-96 konfigurasi probability-family/decoder menghasilkan2.208 evaluasi pada23 development scenarios. Mean spellF1 provider-balanced adalah objective; tidak membuang short GT.
- Pengganti kontrol wave4 harus naik minimal0.01, menang minimal2/3 provider, tanpa provider turun lebih dari0.03. Bila tidak ada, kontrol dipertahankan; best raw alternative tetap dilaporkan.
- Holdout lama sudah terekspos.6 paired SkillCorner ball-drop stress dibuat setelah freeze; bukan6 match baru. Holdout16 scenarios berasal dari9 match unik.
- Hanya empat role fixed dievaluasi publik setelah refit/rebuild: selected,best_new,previous_wave4,pure_boundary. Role dengan hash sama bukan hasil independen. Tidak ada96-way public sweep.

Source, recipe dan input dihash sebelum training; selection dan checkpoint hashes dibekukan sebelum holdout/stress/public. Perubahan algoritme membutuhkan campaign baru, bukan mengganti hash. `dev` dapat memakai cache pekerjaan identik yang selesai tetapi tidak menimpa selection frozen.

## Artefak

`cache/` menyimpan prediksi, stress packs/features dan frozen GRU stress probabilities. `out/` menyimpan preregistration, ranking, model checkpoints, holdout/short-duration diagnostics, audit publik dan CSV submission. Keduanya ignored. Agregat aman dibagikan di `../results/wave5_*.json`, penjelasan hasil di [RESULTS.md](RESULTS.md).

Tidak ada commit/push otomatis. Source dan eksperimen anggota lain tetap terpisah.
