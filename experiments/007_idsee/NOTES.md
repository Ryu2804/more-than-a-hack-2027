# 007 - IDSSE Dead-Ball Cues

## Hipotesis

Dead-ball cue (ball out of pitch, player slow) yang dipakai di 002 dan divalidasi di
Metrica (006) bisa diuji pada data IDSSE yang memiliki ground truth ball status.
Ini menguji apakah cue dead-ball transfer ke IDSSE.

## Aset yang dipakai

- Data IDSSE parsed (J03WN1, J03WR9, dll.) dengan ball status ground truth.
- Run: `parse.py`, `calibrate.py`, `cues.py` (urutan eksekusi: parse -> calibrate -> cues).

## Hasil

- `long_missing_125` dan `missing_any`: F1 = 0.0 di J03WN1 (IDSSE). P(dead | ball missing) = 0.
  Oracle-dead harness justified: missing ball bukan dead ball di IDSSE.
- `ball_slow_0p5_25`: P = 0.950, R = 0.192. Presisi tinggi tapi coverage sangat rendah.

## Verdict

**Diagnostic.** Cue dead-ball berbasis missing ball tidak berfungsi di IDSSE (P=0).
Cue berbasis ball speed (slow) memiliki presisi tinggi tapi recall rendah.

## Compliance

- Evaluasi di data IDSSE (pertandingan lain, bukan BAR-ATH). Tidak melanggar aturan.
- Urutan eksekusi: parse.py -> calibrate.py -> cues.py. Tidak terdokumentasi sebelumnya;
  ditambahkan saat audit 2026-11-08.
- Tidak ada NOTES.md sebelumnya; file ini dibuat saat audit 2026-11-08.