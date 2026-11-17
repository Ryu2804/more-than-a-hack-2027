# Peta branch dan status dokumen

Metode final tim pada semua branch adalah **Farras wave5 `short_gru25/dead45`**. Mulai laporan dari [FINAL_METHOD.md](FINAL_METHOD.md) dan [final_method.json](final_method.json). Folder, istilah primary/control, dan nama final pada eksperimen lama tidak mengganti keputusan ini.

| Branch | Peran setelah penyelarasan | Bahan yang bukan deskripsi metode final |
|---|---|---|
| `main` | Pintu masuk repo, implementasi dan CSV wave5 hasil merge Farras, aturan resmi, dan laporan final | Baseline kompetisi tetap referensi, bukan kandidat tim |
| `farras` | Source implementasi dan artefak kandidat terpilih; riwayat riset Farras | Wave1/2, wave6 dan campaign lanjutan tetap eksperimen, bukan pengganti wave5 |
| `ryu` | Arsip riset Ryu dan fondasi PP-TCN/fitur | `TECHNICAL_DOCUMENT.md`, `writeup.md`, `experiments/RESEARCH.md`, dan folder `015_final_pipeline` menjelaskan varian Ryu lama |
| `jeremy` | Arsip riset SetSeq, auxiliary tasks dan recovery | `exp_1` sampai `exp_7` memakai protokol/keputusan campaign sendiri; exp7 tidak diadopsi |
| `rasya` | Arsip riset rule-based touch/dead cues | `TECHNICAL.md` dan `CONTEXT.md` bukan laporan teknis final wave5 |
| `consolidation` | Arsip integrasi/adapters dan pembanding historis | `CANDIDATES.md`, `docs/repo_inventory.md`, dan common benchmark bukan ranking exact kandidat final |

## Snapshot provenance

Commit di bawah mengidentifikasi sumber riset sebelum penyelarasan dokumentasi. Ini **bukan** daftar HEAD terbaru setelah commit dokumentasi. Histori branch tidak di-rebase, di-reset, di-force-push, atau diganti dengan branch anggota lain.

| Branch | Commit sumber riset |
|---|---|
| main | `ef22d025b0c02bcdbcec53df34973ece0f53a701` |
| farras | `7254990d82ce5dec3bf1fff6baa4fe2f639e3a60` |
| ryu | `5eaf3e595e91fa0502aac48037d0cae8c356ec6f` |
| jeremy | `ccd7b8c5f579b4c7074f20fe75ac1135c99c4bb1` |
| rasya | `a13249724b994728b5686d01bc48e6647cd1e40e` |
| consolidation | `264efa67e1a2e74cce22fea81ed784acf159b258` |

## Membaca angka historis dengan benar

- Public diagnostic, provider-balanced development, IDSSE-only proxy, frame statistics, dan private score adalah jenis hasil berbeda. Jangan membuat leaderboard gabungan dari kolom yang berbeda.
- Angka Jeremy exp2/3/5 sekitar 0.408/0.410/0.488 pada registry lama adalah dead-share prediksi target, bukan public spellF1. Bukti public Jeremy yang teridentifikasi berasal dari exp1, bukan otomatis exp7.
- Common benchmark lama memakai Farras wave2 dengan training terbatas, shipped Ryu weights yang sudah melihat IDSSE benchmark, Rasya GT-assisted orientation/X mask pada evaluasi eksternal, serta adapter Jeremy lama. Angka tabel itu tidak menetapkan urutan kualitas exact submission atau kepatuhan aturan.
- Benchmark adil Farras berikutnya melatih ulang keluarga model dengan split bersama, tetapi bukan reproduksi exact final pipeline tiap branch. Keunggulan mean short-loss atas SetSeq belum terpisah jelas oleh intervalnya.
- File hasil campaign tetap dipertahankan sebagai bukti, termasuk hasil negatif. Label arsip berarti tidak boleh dipakai sebagai spesifikasi final, bukan berarti semua hasil di dalamnya salah.

## Alur kerja laporan

Gunakan identitas, parameter, cohort metric, atribusi, dan batas klaim dari dokumen final. Bila fakta tidak tercantum atau belum diverifikasi, tandai belum diketahui; jangan melengkapinya dari nama folder atau skor maksimum. Perubahan angka atau metode harus merujuk artefak dan hash yang baru, bukan mengganti label pada hasil lama.

Branch `main` dan `farras` memuat paket CSV serta exporter wave5. Branch penelitian lain merujuk metode yang sama tanpa memindahkan atau mengubah algoritme eksperimennya. Sumber implementasi yang dikunci untuk provenance tetap release Farras `7254990d82ce5dec3bf1fff6baa4fe2f639e3a60`.

Integrasi Farras ke `main` diverifikasi dengan exporter dan validator resmi: 199.319 baris, hash identik dengan kandidat frozen, tanpa `annotation_harness`. [Bukti integrasi](merge_verification.json). `report_verification.json` merekam penyelarasan dokumentasi sebelum integrasi ini, bukan hash peta branch yang telah diperbarui.
