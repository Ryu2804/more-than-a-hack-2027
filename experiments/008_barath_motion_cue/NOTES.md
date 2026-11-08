# 008 - BAR-ATH Motion Cue Dead (Public Window Diagnostic)

## Hipotesis

Motion cue (median player speed) bisa mendeteksi dead ball di BAR-ATH, menambah
evidence yang ada.

## Aset yang dipakai

- Data BAR-ATH (public window only, 7500 frame).
- Run: `run.py`.

## Hasil (public window, diagnostik)

| Arm | lambda | spellF1 |
|-----|--------|---------|
| evidence_final | 32 | 0.4875 |
| motion | 8 | 0.020 |
| union | 8 | 0.021 |

Motion cue tidak membantu di BAR-ATH window.

## Compliance — Catatan Audit

- **Lambda selection**: `pick_lambda()` memilih lambda berdasarkan `COUNT_BAND` dan
  `COUNT_CENTRE`, yang berasal dari count band yang dijustifikasi dengan data publik.
  Ini berarti lambda TIDAK dipilih secara independen dari data publik. Meskipun
  hasil dicatat sebagai "diagnostic only", pemilihan lambda sendiri menggunakan
  konstanta yang diturunkan dari data publik. Ini adalah pelanggaran lembut terhadap
  EXPERIMENT_RULES yang melarang tuning pada public window.
- **Evaluasi hanya di public window**: tidak ada uji di IDSSE. Ini diagnostic,
  boleh, tapi tidak bisa dipakai untuk seleksi dan tidak membuktikan transfer.
- Tidak ada pra-registrasi untuk eksperimen ini.
- Status yang tepat: **Exploratory / diagnostic**, bukan "supported".

## Verdict

**Not supported.** Motion cue tidak membantu di BAR-ATH window.

## Menjalankan

```
python experiments/008_barath_motion_cue/run.py --data-dir ../data --out-dir out
```