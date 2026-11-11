"""
Phase 2 — Audit sebaran gambar & kelas per site (dumpsite) DroneWaste.
Dipakai untuk merancang split train/val/test berbasis site (anti-leakage).

Jalankan:  python -m src.cv.audit_sites
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "raw" / "dronewaste" / "dronewaste_v1.0.json"
OUT_M = ROOT / "reports" / "metrics"


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    coco = json.loads(ANN.read_text(encoding="utf-8"))

    img = pd.DataFrame(coco["images"])[["id", "site"]]
    ann = pd.DataFrame(coco["annotations"])[["image_id", "category_id"]]
    cats = (pd.DataFrame(coco["categories"])[["id", "name"]]
            .rename(columns={"id": "category_id"}))
    ann = ann.merge(img, left_on="image_id", right_on="id").merge(cats, on="category_id")

    # --- Ringkasan per site ---
    site = img.groupby("site").agg(n_images=("id", "size"))
    site["n_annotated_images"] = ann.groupby("site")["image_id"].nunique()
    site["n_annotations"] = ann.groupby("site").size()
    site["n_classes"] = ann.groupby("site")["name"].nunique()
    site = site.fillna(0).astype(int).sort_values("n_annotations", ascending=False)

    print("=" * 60)
    print(f"1) JUMLAH SITE: {len(site)}")
    print(site.to_string())
    site.to_csv(OUT_M / "dronewaste_per_site.csv")

    # --- Kelas x site ---
    mat = pd.crosstab(ann["name"], ann["site"])
    summary = pd.DataFrame({
        "total_annotations": mat.sum(axis=1),
        "n_sites_present": (mat > 0).sum(axis=1),
    }).sort_values("total_annotations", ascending=False)

    print("\n2) KELAS: total anotasi & muncul di berapa site")
    print(summary.to_string())
    mat.to_csv(OUT_M / "dronewaste_class_by_site.csv")
    print(f"\nMatriks lengkap kelas x site → {OUT_M / 'dronewaste_class_by_site.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()