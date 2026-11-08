# 009 - Ownership Accuracy per Distance/Speed Bin (IDSSE J03WOH)

## Hipotesis

Ownership accuracy (nearest player) meningkat pada jarak dekat dan bola lambat.
Gate jarak < 1m bisa meningkatkan akurasi emission DP.

## Aset yang dipakai

- Data IDSSE parsed (J03WOH) dengan ball status dan possession ground truth.
- Run: `run.py`.

## Hasil (IDSSE J03WOH, diagnostik)

| Bin | Accuracy | Coverage |
|-----|----------|----------|
| distance [0.0, 0.5m) | 0.916 | 12.4% |
| distance [0.5, 1.0m) | 0.919 | 25.6% |
| distance [1.0, 2.0m) | 0.787 | -- |
| ball_speed >= 8 km/h | 0.585 | -- |

Gate jarak < 1m: coverage ~38%, akurasi ~0.918.

Namun 010 menunjukkan gate ini MENURUNKAN spellF1 (0.375 vs 0.4875).
Mekanisme: gating mengurangi coverage, DP mengisi gap dengan carry.

## Verdict

**Diagnostic.** Gate meningkatkan per-frame ownership accuracy tapi menurunkan spellF1.
Trade-off ini belum dijelaskan secara formal.

## Compliance

- Evaluasi di data IDSSE (pertandingan lain, bukan BAR-ATH). Tidak melanggar aturan.
- Tidak ada NOTES.md sebelumnya; file ini dibuat saat audit 2026-11-08.