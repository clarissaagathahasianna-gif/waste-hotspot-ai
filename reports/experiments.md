# Experiment Log — Computer Vision (DroneWaste, site-level split)

**Dataset:** DroneWaste (Zenodo, CC BY 4.0), 20 kelas → 16 kelas (5 kelas langka digabung menjadi `Other waste`).
**Split per site (anti-leakage):**
- Train: 12 site (961 gambar, termasuk 144 background hasil sampling)
- Val: site5, site15 (353 gambar)
- Test: site2, site7, site13 (305 gambar, 768 instance)

**Setelan umum:** imgsz=640, batch=16, epochs=100, patience=20, seed=42, deterministic=True.
Satu run per eksperimen. Semua metrik di bawah dihitung pada **TEST set**.

## Hasil

### Mask (segmentasi)
| ID | Model | Perubahan | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Baseline | yolo11n-seg | 16 kelas, default | 0.349 | 0.216 | 0.186 | 0.121 |
| E1 | yolo11n-seg | single_cls=True | 0.398 | 0.439 | 0.344 | 0.199 |
| E2 | yolo11n-seg | flipud=0.5, degrees=45 | 0.323 | 0.277 | 0.219 | 0.142 |
| E3 | yolo11s-seg | sama dengan E2 | 0.206 | 0.322 | 0.215 | 0.136 |
| E4 | yolo11n-seg | single_cls + flipud=0.5, degrees=45 | 0.422 | 0.441 | 0.346 | 0.194 |

### Box (deteksi)
| ID | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Baseline | — | — | — | — |
| E1 | 0.425 | 0.469 | 0.382 | 0.242 |
| E2 | 0.326 | 0.284 | 0.235 | 0.160 |
| E3 | 0.209 | 0.325 | 0.227 | 0.151 |
| E4 | 0.438 | 0.461 | 0.383 | 0.239 |

> Box baseline belum dicatat — isi dari output Sel 6 notebook `05-train-yolo11-seg-baseline`.

## Temuan

**Error analysis baseline (confusion matrix test):**
- Kelas tumpukan (*pile_waste*) saling tertukar; `Mixed items` menjadi "keranjang sampah" tebakan
  (Excavation 57%, Other waste 50%, Plastic 39% diprediksi sebagai Mixed items).
- Banyak objek terlewat sebagai background: Rubble 47%, Metal barrels 42%.
- Kelas berbentuk khas terdeteksi baik: Tyres 94%, Asbestos 86%, Pallets 85%, Textile 68%.
- Kurva training: train loss terus turun, val loss mendatar sejak ±epoch 15 dan val/seg_loss naik
  ±epoch 50 → awal overfitting; gap val/cls_loss (~5) vs train/cls_loss (~1) menunjukkan domain gap antar-site.

**Perbandingan eksperimen:**
- **E2 vs Baseline:** augmentasi udara cenderung membantu model 16 kelas
  (recall 0.216 → 0.277; mask mAP50 0.186 → 0.219).
- **E3 vs E2:** model lebih besar (yolo11s) tidak membantu → hambatan utama ada di data
  (jumlah, variasi antar-site, label tumpukan yang ambigu), bukan kapasitas model.
- **E1:** model 1 kelas jauh lebih andal untuk mendeteksi *keberadaan* sampah
  (mask mAP50 0.344, recall 0.439).
  Catatan: mAP 1 kelas tidak setara langsung dengan mAP rata-rata 16 kelas
  (rata-rata 16 kelas ikut tertarik turun oleh kelas langka).
- **E4 vs E1:** augmentasi tidak terbukti membantu model 1 kelas
  (mask mAP50-95 0.194 vs 0.199; selisih jauh di bawah variasi acak satu run).

## Keputusan final
| Peran di pipeline | Model | Alasan |
|---|---|---|
| Detektor sampah (keberadaan & luas) | **E1** | Setara E4, konfigurasi lebih sederhana |
| Identifikasi material (mis. asbestos) | **E2** | Model 16 kelas terbaik; hanya beberapa kelas andal (Asbestos, Pallets, Tyres, Textile) |

Bobot model:
- `models/experiments/runs/E1_y11n_singlecls/weights/best.pt`
- `models/experiments/runs/E2_y11n_aerialaug/weights/best.pt`

Tuning Computer Vision dihentikan di sini; fokus proyek beralih ke prediksi hotspot.

## Keterbatasan
- Test set kecil (3 site, 768 instance); beberapa kelas hanya 1–7 instance di test
  (Vehicles, Furniture, Other waste, Excavation materials) → AP per kelas tersebut tidak andal.
- Satu seed per eksperimen; selisih kecil antar-eksperimen bisa disebabkan variasi acak.
- Domain data: dumpsite Eropa; model belum diuji pada citra udara Indonesia.