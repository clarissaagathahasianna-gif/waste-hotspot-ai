"""
Phase 2 — Split train/val/test berbasis SITE (anti-leakage) untuk DroneWaste.
- Menggabungkan kelas langka menjadi 'Other waste' (20 -> 16 kelas)
- Mencari kombinasi site test/val terbaik secara otomatis (exhaustive search)
- Menyimpan hasil ke configs/split_sites.json

Jalankan:  python -m src.cv.make_site_split
"""
import itertools
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "raw" / "dronewaste" / "dronewaste_v1.0.json"
CONFIGS = ROOT / "configs"
OUT_M = ROOT / "reports" / "metrics"

RARE_CLASSES = {"Appliances", "Paper", "Electronic equipment", "Foundry", "Asphalt milling"}
OTHER = "Other waste"
TEST_FRAC = (0.12, 0.22)   # porsi anotasi untuk test
VAL_FRAC = (0.10, 0.20)    # porsi anotasi untuk val
TARGET = 0.15              # porsi ideal


def load():
    coco = json.loads(ANN.read_text(encoding="utf-8"))
    img = pd.DataFrame(coco["images"])[["id", "site"]]
    cats = (pd.DataFrame(coco["categories"])[["id", "name"]]
            .rename(columns={"id": "category_id"}))
    ann = (pd.DataFrame(coco["annotations"])[["image_id", "category_id"]]
           .merge(img, left_on="image_id", right_on="id")
           .merge(cats, on="category_id"))
    ann["cls"] = ann["name"].where(~ann["name"].isin(RARE_CLASSES), OTHER)
    class_map = {n: (OTHER if n in RARE_CLASSES else n) for n in cats["name"]}
    return img, ann, class_map


def search(ann):
    mat = pd.crosstab(ann["site"], ann["cls"])      # site x kelas
    sites, M = list(mat.index), mat.to_numpy()
    total, n_cls = M.sum(), M.shape[1]
    idx = {s: i for i, s in enumerate(sites)}

    def stats(group):
        v = M[[idx[s] for s in group]].sum(axis=0)
        return v.sum() / total, int((v > 0).sum())  # (porsi anotasi, jumlah kelas)

    best = None
    for test in itertools.combinations(sites, 3):
        f_t, c_t = stats(test)
        if not TEST_FRAC[0] <= f_t <= TEST_FRAC[1]:
            continue
        rest = [s for s in sites if s not in test]
        for k in (2, 3):
            for val in itertools.combinations(rest, k):
                f_v, c_v = stats(val)
                if not VAL_FRAC[0] <= f_v <= VAL_FRAC[1]:
                    continue
                train = [s for s in rest if s not in val]
                if stats(train)[1] < n_cls:        # train wajib punya semua kelas
                    continue
                key = (c_t + c_v, -(abs(f_t - TARGET) + abs(f_v - TARGET)))
                if best is None or key > best[0]:
                    best = (key, list(test), list(val), train)
    return best, list(mat.columns)


def report(img, ann, split):
    rows = []
    for name, sites in split.items():
        im, an = img[img["site"].isin(sites)], ann[ann["site"].isin(sites)]
        rows.append({
            "split": name, "n_sites": len(sites), "n_images": len(im),
            "n_annotated_images": an["image_id"].nunique(),
            "n_annotations": len(an),
            "pct_annotations": round(len(an) / len(ann) * 100, 1),
            "n_classes": an["cls"].nunique(),
        })
    summary = pd.DataFrame(rows).set_index("split")
    site2split = {s: k for k, v in split.items() for s in v}
    per_cls = pd.crosstab(ann["cls"], ann["site"].map(site2split))[["train", "val", "test"]]
    return summary, per_cls


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OUT_M.mkdir(parents=True, exist_ok=True)

    img, ann, class_map = load()
    print(f"Kelas setelah penggabungan: {ann['cls'].nunique()}")

    best, classes = search(ann)
    if best is None:
        print("Tidak ada kombinasi yang memenuhi syarat. Longgarkan TEST_FRAC/VAL_FRAC.")
        return
    _, test, val, train = best
    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    summary, per_cls = report(img, ann, split)
    print("\n1) SITE PER SPLIT")
    for k, v in split.items():
        print(f"   {k:5s}: {', '.join(v)}")
    print("\n2) RINGKASAN SPLIT")
    print(summary.to_string())
    print("\n3) ANOTASI PER KELAS PER SPLIT")
    print(per_cls.to_string())
    for k in ("val", "test"):
        missing = per_cls.index[per_cls[k] == 0].tolist()
        print(f"\n   Kelas TIDAK ada di {k}: {missing if missing else '-'}")

    out = {
        "strategy": "site-level grouped split (exhaustive search)",
        "criteria": {"test_frac": TEST_FRAC, "val_frac": VAL_FRAC, "target": TARGET},
        "classes": classes,
        "class_map": class_map,
        **split,
    }
    (CONFIGS / "split_sites.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    per_cls.to_csv(OUT_M / "split_class_counts.csv")
    print(f"\nTersimpan → configs/split_sites.json & reports/metrics/split_class_counts.csv")


if __name__ == "__main__":
    main()