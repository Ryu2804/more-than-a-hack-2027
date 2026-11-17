# Penguasaan Bola: design handoff

## Status

Arsip handoff untuk fase desain FigCraft. File desain dan preview di folder ini tetap merekam UI bahasa Indonesia pada fase tersebut. Aplikasi web sudah diimplementasikan dalam bahasa Inggris; gunakan [README web](../README.md) untuk status implementasi, verifikasi browser, dan deployment. Dokumen teknis serta kode penelitian di luar `web/` tidak diubah.

Arah visual tetap mengikuti [brief](../football_match_ball_possession_dashboard_ui_ux_spec.md): light workspace, Inter, canvas `#F6F7F9`, kartu putih, biru Barcelona, hijau Athletic Club, amber bola mati. ENERGY 2 / RHYTHM 2 / MOTION 1.

## Cakupan tampilan

Semua ringkasan dan grafik sekarang memakai **5 menit yang memiliki label acuan publik**, bukan seluruh pertandingan.

- Acuan batas: [metadata publik](../../evaluation/gt/BAR-ATH_possession_gt_public_meta.json).
- Sumber video tactical: frame 11185 sampai 18684, inklusif.
- Interval setengah terbuka: `[11185, 18685)`, 7.500 frame pada 25 fps.
- Bagian file asli: `07:27.40` sampai `12:27.40`, batas akhir eksklusif.
- Waktu yang terlihat: `00:00` sampai `05:00`, dihitung dari awal cuplikan.
- Bila memakai cuplikan terpotong: `source_frame = 11185 + clip_frame`.
- Bila membuka file pertandingan asli: batasi pemutaran/seek pada `[447.4, 747.4)` detik. Jangan menampilkan bagian lain sebagai bagian evaluasi publik.
- Kamera broadcast tidak punya pemetaan waktu yang diverifikasi; jangan menganggap waktunya sama dengan tactical.

Detail yang diperlukan implementasi tetap berada di metadata ini dan [games.py](../../games.py), bukan di teks antarmuka.

## Perbandingan yang adil

[Label acuan publik](../../evaluation/gt/BAR-ATH_possession_gt_public.csv) dan [hasil model](../../submission/artifacts/farras_wave5.csv) dipotong pada interval yang identik. Dua baris di Replay dan MobileReplay memakai titik awal, lebar, warna, dan skala waktu yang sama:

1. **Label acuan**: 21 run label, termasuk 17 penguasaan Barcelona/Athletic Club.
2. **Hasil model**: 15 run label, termasuk 12 penguasaan Barcelona/Athletic Club.

Segmen digambar langsung dari run-length encoding setiap sumber. Tidak ada smoothing, koreksi label, pergeseran waktu, pelatihan, pemilihan model, atau penghitungan ulang skor. GT publik dibaca hanya untuk perbandingan visual yang diminta pengguna. GT privat tidak digunakan.

Lihat [replay.json](replay.json) untuk kontrak interval dan sumber; [evidence.json](evidence.json) untuk distribusi yang ditampilkan. Jumlah prediksi pada cuplikan: Barcelona 3.585, Athletic Club 2.860, bola mati 1.055, tidak diketahui 0. Persentase memakai denominator 7.500. `55,6%` memakai denominator Barcelona + Athletic Club dan diberi keterangan “Barcelona saat bola aktif”.

Ringkasan area/garis memakai 30 interval masing-masing 250 frame atau 10 detik. Nilainya frekuensi label nyata, digambar sebagai step, bukan probabilitas atau spline.

## Bahasa dan skor

Nama metode final dan rincian riset tetap tersedia di [sumber laporan](../../docs/final_method.json); tidak ditampilkan dalam UI.

Skor terpublikasi hanya diubah skala tampilannya, bukan dihitung ulang:

| Label UI | Sumber | Tampilan |
| --- | --- | --- |
| Penguasaan bola / 100 | Public spellF1 0.7724137931034483 | 77,2 |
| Tiap momen / 100 | Public macroF1 0.9020586271571484 | 90,2 |
| Pertandingan lain / 100 | External-development spellF1 0.670099305497763 | 67,0 |

Ini **skor dari 100, bukan persentase akurasi**. Nilai pertandingan lain hanya ada pada halaman penjelasan dan dipisahkan dari penilaian cuplikan. Kartu Replay berisi dua skor publik serta jumlah penguasaan acuan/hasil model.

Batas penilaian tetap dijelaskan dengan bahasa biasa: cuplikan sudah pernah dilihat selama pengembangan, pengujian baru masih diperlukan, hasil seluruh pertandingan belum diketahui, dan pembuatan ulang dari awal belum lengkap.

## Video lokal dan gambar contoh

Cuplikan `cuplikan-acuan-5-menit.mp4` dibuat secara lokal **di luar repo**, berdurasi 300 detik, 7.500 frame, 1920 × 1080. Frame pertama dan terakhir dapat didekode. Frame pertama paling dekat dengan frame sumber 11185 saat dibandingkan dengan tetangganya; perbedaan kompresi tetap ada.

Jangan menyalin atau memublikasikan cuplikan, video asli, maupun frame dari bundle ke GitHub, Vercel, atau dokumen publik. Aturan [RULES.md](../../RULES.md) melarang redistribusi data dan clipped video. Implementasi web menggunakan file lokal; revoke object URL saat file diganti.

Poster di FigCraft masih merupakan potongan gambar yang diberikan pengguna, dengan kredit [Hadi Sotudeh / LinkedIn](https://lnkd.in/p/gDmRew2e). Label “Gambar contoh” tetap terlihat. Poster bukan bukti video sedang diputar atau sesuai dengan momen yang ditunjuk; izin reuse production belum diverifikasi. Tidak ada GT, posisi pemain, atau skor model referensi yang disalin dari screenshot.

## Layar dan interaksi

Sepuluh frame tetap tersedia: Desktop, Tablet, Mobile, Evidence, MobileEvidence, States, Replay, MobileReplay, MediaSetup, MobileMediaSetup. File menyimpan 6 flow dan 39 koneksi prototype. Ikon transport adalah contoh tombol, bukan pemutar aktif.

Grid overview desktop tetap 2 × 2, gap 20, card 638 × 356. Mobile memakai satu kolom; ringkasan dan perbandingan tetap terbaca. Target sentuh minimum 44 px, font minimum 11 px, focus ring 2 px, dan setiap warna disertai nama tim/keadaan bola. Teks di atas poster memakai bidang gelap buram.

Saat implementasi: satu cursor relatif cuplikan memperbarui video dan dua baris label bersama. Jangan membiarkan hasil dari momen lama tampil seolah cocok setelah seek. Tampilkan belum memilih video, buffering, seek, decode error, dan akhir cuplikan secara jelas. Kegagalan video tidak boleh menghilangkan perbandingan label. Jangan membuat posisi pemain dari label penguasaan.

## Implementasi berikutnya

Bagian implementasi pada handoff ini bersifat historis. Monorepo, aplikasi, lockfile, build produksi, dan pemutaran video lokal sekarang tersedia. Status deployment serta verifikasi aplikasi mengikuti [README web](../README.md), bukan hasil pemeriksaan desain statis di folder ini.
