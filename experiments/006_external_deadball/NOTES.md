# 006 - External Dead-Ball Benchmark (Metrica)

## Hipotesis

Dead-ball cue (ball out of pitch, player slow) yang dipakai di 002 bisa divalidasi pada
data Metrica Sample_Game, yang memiliki event ground truth restart. Ini menguji apakah
cue dead-ball transfer ke pertandingan lain.

## Aset yang dipakai

- Metrica Sample_Game_1 dan Sample_Game_2 (tracking + events).
- Run: `run.py` dengan nan_run_25 dan nan_run_125.

## Hasil

| Arm | F1 | Precision | Recall |
|-----|----|-----------|--------|
| nan_run_25 | 0.450 | 0.306 | 0.853 |
| nan_run_125 | 0.443 | 0.303 | 0.829 |

Gating dengan restart MENURUNKAN F1: recall drop lebih besar dari precision gain.

Temuan kritis: P(dead | ball missing) = 0.305 di Metrica. Missing != dead.
Ini yang melatarbelakangi oracle-dead harness di IDSSE (007).

## Verdict

**Exploratory.** Cue dead-ball memiliki precision rendah (~0.30) tapi recall tinggi (~0.85)
di Metrica. Gating dengan restart tidak membantu.

## Compliance

- Evaluasi di data Metrica (pertandingan lain, bukan BAR-ATH). Tidak melanggar aturan.
- Lisensi Metrica sample: belum dikonfirmasi (GitHub license API: null). Jika Metrica
  dipakai di submission, lisensi harus dikonfirmasi sebelum commit.
- Tidak ada NOTES.md sebelumnya; file ini dibuat saat audit 2026-11-08.

## Menjalankan

```
python experiments/006_external_deadball/run.py --data-dir ../data --out-dir out
```