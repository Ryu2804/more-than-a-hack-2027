# 003 - Possession layers (ownership, identity, Smart Tagging, duration)

Hypothesis: the components left untried in 002 add spellF1 on top of `dp_ev` and `ev_smooth125`.
Each family gets its own switch penalty from the GT-free possession-count band, so no layer is
judged with someone else's calibration.

Change vs 002: soft ownership from per-team minimum distance, ball-in-flight neutrality, Smart
Tagging Team prior on attacking phases and restarts, identity repair (pre-DP and post-DP), and a
duration prior through iterative reweighting.

Verdict: exploratory, no layer proven. Soft ownership ~ flat vs `dp_ev`; Team prior and duration
prior slightly worse on the public window and not separable; identity repair inert at the selected
penalty (0 frames changed post-DP). `ev_smooth125` remains the public-window leader. Full report:
`../research_loop/riset_methodology/experiments/possession_layers_v3/REPORT.md`.

## Compliance — Catatan Audit

- **Tabel angka numerik per arm tidak ada di NOTES.md.** Hasil per arm (spellF1, CI, lambda)
  hanya ada di `out/results.json` dan laporan lengkap di `research_loop/`. Untuk reproducibility,
  seharusnya ada tabel ringkasan di sini. Ditambahkan saat audit 2026-11-08.
- Semua evaluasi di public window (diagnostik). Tidak ada pra-registrasi.
- Tidak ada kandidat yang dipilih dari eksperimen ini.

Run (from repo root):

```
.venv/bin/python experiments/003_possession_layers/run.py --data-dir ../data --out-dir out --bootstrap 1000 --synthetic-seeds 3
```
