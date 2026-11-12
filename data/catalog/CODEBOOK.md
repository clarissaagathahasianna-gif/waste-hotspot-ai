# Codebook — Katalog Insiden Pembuangan Sampah Bandung–Cimahi

## Kolom
| Kolom | Isi | Contoh |
|---|---|---|
| incident_id | ID unik berurutan | INC001 |
| site_key | ID lokasi; sama jika lokasi yang sama muncul di berita lain | gunung_batu_tps |
| date_published | Tanggal berita terbit (YYYY-MM-DD) | 2025-11-14 |
| date_event | Tanggal kejadian/pantauan jika disebut; kosongkan jika tidak | 2025-11-13 |
| location_text | Lokasi persis seperti tertulis di berita | TPS Gunung Batu Timur |
| kelurahan / kecamatan | Diisi saat geocoding | |
| city | Kota Bandung / Kota Cimahi | Kota Bandung |
| lat, lon | Koordinat desimal (WGS84), 6 angka di belakang koma | -6.912345 |
| geocode_precision | exact (titik jelas) / street (ruas jalan) / kelurahan / kecamatan | street |
| incident_type | tps_liar / tps_overload / pinggir_jalan / sungai / pasar / lainnya | pasar |
| is_official_tps | yes / no / unknown | no |
| source_name | Nama media | detikJabar |
| source_url | Link berita | https://... |
| notes | Catatan (volume, durasi menumpuk, dll.) | menumpuk 5 bulan |

## Aturan
1. Satu baris = satu lokasi dalam satu berita. Lokasi yang sama di berita lain → baris baru, `site_key` sama.
2. Hanya media berita (bukan unggahan media sosial).
3. Wilayah: hanya Kota Bandung & Kota Cimahi.
4. Geocoding pakai openstreetmap.org (klik kanan → "Show address" → koordinat muncul di panel kiri).
5. Hanya baris dengan precision `exact` atau `street` yang dipakai sebagai label model (grid 500 m).
6. Jangan menebak koordinat. Jika lokasi tidak bisa dipastikan, isi precision `kelurahan`/`kecamatan`.
7. Kolom kelurahan & kecamatan boleh dikosongkan; diisi otomatis oleh script dari batas administrasi.
8. Target penelitian = titik penumpukan sampah bermasalah (TPS liar + TPS resmi yang meluap).
   Kolom is_official_tps dipakai untuk analisis sensitivitas.

## Log pencarian
| Tanggal cari | Query | Media | Jumlah artikel dicek |
|---|---|---|---|