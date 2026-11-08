# 010 - Ownership Confidence Gate

## Hipotesis

Gate jarak < 1m + ball speed < 2 m/s meningkatkan ownership accuracy,
sehingga emission DP menjadi lebih baik.

## Aset yang dipakai

- Data BAR-ATH (public window, 7500 frame).
- Run: `run.py`.

## Hasil (public window, diagnostik)

| Arm | lambda | spellF1 |
|-----|--------|---------|
| dp_atd_nearest | 32 | 0.4875 |
| dp_gated (gate d<1m) | 8 | 0.375 |

Verdict: **Not supported.** Gate MENURUNKAN spellF1.

## Compliance — Catatan Audit

- **Lambda selection**: `pick_lambda()` memilih lambda berdasarkan `COUNT_BAND` dan
  `COUNT_CENTRE` (turunan dari data publik). Lambda 8 untuk dp_gated berbeda dari
  lambda 32 baseline, menunjukkan ada pemilihan. Jika lambda dipilih dari grid di
  public window, ini melanggar EXPERIMENT_RULES.
- **verdict field kosong** di results.json ("?"). Tidak ada rekaman formal verdict.
- Tidak ada pra-registrasi untuk eksperimen ini.
- Status yang tepat: **Not supported**, bukan "inconclusive".

## Verdict

**Not supported.** Gate menurunkan spellF1 (0.375 vs 0.4875).

## Menjalankan

```
python experiments/010_ownership_confidence_gate/run.py --data-dir ../data --out-dir out
```