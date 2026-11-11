"""
Phase 2 — Audit dataset DroneWaste.
Cek: struktur JSON, jumlah gambar/anotasi, nama 20 kelas, distribusi kelas,
ukuran gambar, file yang hilang, dan metadata tambahan (mis. info lokasi/site).

Jalankan dari ROOT proyek:
    python -m src.cv.audit_dronewaste
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DW = ROOT / "data" / "raw" / "dronewaste"
ANN = DW / "dronewaste_v1.0.json"
OUT_M = ROOT / "reports" / "metrics"
OUT_F = ROOT / "reports" / "figures"
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)

    coco = json.loads(ANN.read_text(encoding="utf-8"))
    images, anns, cats = coco["images"], coco["annotations"], coco["categories"]

    print("=" * 60)
    print("1) STRUKTUR JSON")
    print("   Kunci utama      :", list(coco.keys()))
    print("   Kolom per gambar :", list(images[0].keys()))
    print("   Kolom per anotasi:", list(anns[0].keys()))
    print("   Kolom per kelas  :", list(cats[0].keys()))

    print("\n2) JUMLAH")
    print(f"   Gambar   : {len(images):,}")
    print(f"   Anotasi  : {len(anns):,}")
    print(f"   Kelas    : {len(cats)}")

    # --- Distribusi kelas ---
    ann_per_cat = Counter(a["category_id"] for a in anns)
    img_per_cat: dict[int, set] = {}
    for a in anns:
        img_per_cat.setdefault(a["category_id"], set()).add(a["image_id"])

    cat_df = pd.DataFrame(cats)
    cat_df["n_annotations"] = cat_df["id"].map(ann_per_cat).fillna(0).astype(int)
    cat_df["n_images"] = cat_df["id"].map(lambda i: len(img_per_cat.get(i, ()))).astype(int)
    cat_df = cat_df.sort_values("n_annotations", ascending=False)

    print("\n3) KELAS & DISTRIBUSI (urut terbanyak)")
    print(cat_df.to_string(index=False))
    cat_df.to_csv(OUT_M / "dronewaste_class_distribution.csv", index=False)

    # --- Gambar tanpa anotasi ---
    annotated = {a["image_id"] for a in anns}
    n_empty = sum(1 for im in images if im["id"] not in annotated)
    print(f"\n4) Gambar tanpa anotasi (background): {n_empty:,}")

    # --- Ukuran gambar ---
    sizes = Counter((im.get("width"), im.get("height")) for im in images)
    print("\n5) Ukuran gambar terbanyak (lebar x tinggi : jumlah)")
    for (w, h), n in sizes.most_common(5):
        print(f"   {w} x {h} : {n:,}")

    # --- Ukuran objek (bbox) ---
    areas = pd.Series([a["bbox"][2] * a["bbox"][3] for a in anns])
    print("\n6) Luas bounding box (piksel^2)")
    print(f"   min={areas.min():,.0f}  median={areas.median():,.0f}  max={areas.max():,.0f}")
    print(f"   objek kecil (<32x32 px): {(areas < 32*32).mean():.1%}")

    # --- Cek file di disk ---
    on_disk = {p.name for p in DW.rglob("*") if p.suffix.lower() in IMG_EXT}
    missing = [im["file_name"] for im in images if Path(im["file_name"]).name not in on_disk]
    print(f"\n7) File gambar di disk: {len(on_disk):,} | hilang dari JSON: {len(missing):,}")
    if missing:
        print("   contoh yang hilang:", missing[:3])

    # --- Grafik distribusi kelas ---
    name_col = "name" if "name" in cat_df.columns else cat_df.columns[1]
    plot_df = cat_df.sort_values("n_annotations")
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df[name_col].astype(str), plot_df["n_annotations"])
    plt.xlabel("Jumlah anotasi")
    plt.title("DroneWaste - distribusi anotasi per kelas")
    plt.tight_layout()
    fig_path = OUT_F / "dronewaste_class_distribution.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nGrafik disimpan → {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()