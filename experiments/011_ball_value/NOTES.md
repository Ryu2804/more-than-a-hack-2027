# 011 - Ball Value Study

## Hipotesis

Mengetahui posisi bola dengan sempurna meningkatkan ownership accuracy.
Ball bridging (imputation) bisa mendekati oracle performance.

## Aset yang dipakai

- Data IDSSE parsed dengan ball position ground truth.
- Run: `phase0.py`, `phase0b.py`, `phase1.py`, `phase2.py`, `phase2b.py`, `phase2c.py`.

## Hasil

### Phase 1 (oracle ball, IDSSE)

| Rule | Accuracy | Coverage |
|------|----------|----------|
| gate_1m | 0.939 | 42.5% |
| nearest (full) | 0.808 | 100% |

Ceiling ownership accuracy dengan bola sempurna: ~0.939 pada 42.5% frame.

### Phase 2 (ball_recovered dengan Gaussian imputation)

- ball_masked confirm: spellF1 ~0.636 (J03WN1) dan ~0.595 (J03WR9)
- ball_bridged_k15_s0.5_p0.4 dev: spellF1 ~0.845-0.872

## Compliance — Catatan Audit

- **`candidate_selection` / `worst_passing`**: Fungsi ini adalah **diagnostik**, bukan
  mekanisme seleksi. `worst_passing` hanya dicetak ke stdout, tidak dipakai untuk
  memilih arm yang masuk confirm. Nama `selection_dev_frame` dan `selection_dev_run`
  di results.json misleading — seharusnya `diagnostic_dev_frame` dan `diagnostic_dev_run`.
  Rename akan dilakukan di versi berikutnya.
- **Grid besar tanpa pra-registrasi**: sigma x p x K grid dijalankan dan best dipilih
  dari dev. Tanpa pre-registration, ini post-hoc selection. Namun karena `candidate_selection`
  hanya diagnostik (tidak memilih kandidat), pelanggaran lebih ringan.
- Evaluasi di data IDSSE (pertandingan lain). Tidak melanggar aturan pemisahan data.

## Verdict

**Diagnostic.** Oracle ball memberikan ceiling ~0.939 accuracy. Ball bridging
belum dicoba sebagai input ke touch-hold pipeline di BAR-ATH.

## Menjalankan

```
python experiments/011_ball_value/phase1.py --parsed-dir ../data/external/idsee/parsed --out-dir out
python experiments/011_ball_value/phase2.py --parsed-dir ../data/external/idsee/parsed --out-dir out --split dev
```