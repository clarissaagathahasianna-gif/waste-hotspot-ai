# Experiment Log — Computer Vision (DroneWaste, site-level split)

Test set: site2, site7, site13 (768 instances). Satu run per eksperimen, seed=42.

| ID | Model | Perubahan | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| Baseline | yolo11n-seg | 16 kelas, default | 0.349 | 0.216 | 0.186 | 0.121 |
| E1 | yolo11n-seg | single_cls=True | 0.398 | 0.439 | 0.344 | 0.199 |
| E2 | yolo11n-seg | flipud=0.5, degrees=45 | 0.323 | 0.277 | 0.219 | 0.142 |
| E3 | yolo11s-seg | sama dgn E2 | 0.206 | 0.322 | 0.215 | 0.136 |

## Temuan
- Kesalahan utama baseline: kelas tumpukan tertukar (Mixed items jadi "keranjang sampah"),
  Rubble & Metal barrels banyak terlewat sebagai background.
- Augmentasi udara cenderung membantu (recall 0.216 -> 0.277); belum diuji multi-seed.
- Model lebih besar tidak membantu -> hambatan utama di data, bukan kapasitas model.
- Model 1 kelas lebih andal untuk deteksi keberadaan sampah
  (catatan: mAP 1 kelas tidak setara langsung dengan mAP rata-rata 16 kelas).

## Keterbatasan
- Test set kecil; beberapa kelas hanya 1–7 instance di test (Vehicles, Furniture, Other waste, Excavation).
- Satu seed per eksperimen; selisih kecil bisa karena variasi acak.
- Domain: dumpsite Eropa; belum diuji pada citra Indonesia.

## Rencana
- E4: single_cls + augmentasi udara.