# 005 - Ownership dari warna jersey

## Hipotesis

Warna jersey yang dibaca dari video tactical memberi sinyal tim yang lebih andal daripada label tim
statis pada ATD, sehingga emission DP possession menjadi lebih baik.

## Aset yang dipakai

- `data/tactical/BAR-ATH_tactical.mp4` (199.319 frame, 25 fps) lewat OpenCV, decode lokal.
- `data/tactical/BAR-ATH_tactical_homography.csv` untuk memproyeksikan posisi ATD ke piksel.
- `data/tactical/BAR-ATH_tactical_atd.txt` + `_atd.xml` untuk posisi pemain dan label tim.
- Centroid warna tim dari eksperimen 004 (`out/004_video_team_check/results.json`) dan
  `features.npz` untuk tabel repair per slot.

## Perubahan dari eksperimen sebelumnya

Eksperimen 002/003 memakai emission tim dari pemain terdekat dengan label ATD statis. Di sini
emission diganti atau digerbangi oleh warna jersey, dalam empat bentuk: penggantian per frame,
repair label per slot, gate per slot, gate per frame.

## Hasil (public window, diagnostik)

| Arm | lambda | spellF1 | CI90 | macroF1 | spells | dur (s) | frag |
|---|---|---|---|---|---|---|---|
| dp_atd_nearest | 32 | 0.4875 | 0.3250-0.6500 | 0.6803 | 15 | 17.4 | 1.00 |
| dp_colour_gated | 25 | 0.4000 | 0.2303-0.5697 | 0.6133 | 16 | 16.2 | 1.10 |
| dp_colour_gated_track | 32 | 0.3867 | 0.2267-0.5467 | 0.6497 | 13 | 19.3 | 0.86 |
| dp_track_repaired | 32 | 0.3742 | 0.2065-0.5548 | 0.5697 | 14 | 18.4 | 0.95 |
| dp_colour_obs | 25 | 0.3030 | 0.1570-0.4606 | 0.5260 | 16 | 16.2 | 1.10 |

Delta berpasangan vs `dp_atd_nearest`: gated -0.090 [-0.232,+0.051], gated_track -0.103
[-0.226,+0.009], track_repaired -0.115 [-0.253,+0.015], colour_obs -0.186 [-0.294,-0.096].

## Verdict

**Not supported.** Tidak ada varian warna yang mengalahkan emission ATD; per-frame colour terlalu
bising (149,3 pergantian tim/menit vs 28,4). Temuan yang layak dilanjutkan ada di eksperimen 004:
slot ATD berganti pemain, dan itu yang harus diperbaiki (rekonstruksi identitas track), bukan
label timnya.

## Compliance — Catatan Audit

- **Pernyataan eksplisit**: Hasil public window di atas adalah **diagnostik saja** dan **tidak
  pernah dipakai untuk seleksi** kandidat, lambda, atau konstanta. Lambda 25 dan 32 untuk
  masing-masing arm dipilih berdasarkan count band (turunan dari data publik), bukan dari
  public window spellF1. Namun count band itu sendiri berasal dari data publik, sehingga
  pemilihan lambda tidak sepenuhnya independen. Status: lembut melanggar EXPERIMENT_RULES.
- Tidak ada pra-registrasi untuk eksperimen ini.

## Menjalankan

```
.venv/bin/python experiments/004_video_team_check/run.py --data-dir ../data \
  --out-dir out/004_video_team_check --step 5 --save-visuals
.venv/bin/python experiments/005_possession_colour_owner/run.py --data-dir ../data \
  --out-dir out/005_colour_owner --bootstrap 1000
```

Laporan lengkap: `research_loop/riset_methodology/experiments/possession_colour_v5/REPORT.md`.
