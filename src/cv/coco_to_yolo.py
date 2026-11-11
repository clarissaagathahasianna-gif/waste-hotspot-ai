"""
Phase 3 — Konversi DroneWaste (COCO) -> format YOLO-seg dengan split per site.
- Kelas digabung sesuai configs/split_sites.json (20 -> 16)
- Background di TRAIN di-sampling ulang (val/test tetap utuh)
- Menulis data.yaml, manifest.csv, dan gambar cek label

Jalankan:  python -m src.cv.coco_to_yolo
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "dronewaste"
ANN = RAW / "dronewaste_v1.0.json"
SPLIT = ROOT / "configs" / "split_sites.json"
OUT = ROOT / "data" / "processed" / "yolo_dronewaste"
FIG = ROOT / "reports" / "figures" / "label_check"
OUT_M = ROOT / "reports" / "metrics"

BG_FRAC_TRAIN = 0.15   # porsi gambar background di train setelah sampling
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def polygon_from_ann(a: dict, w: int, h: int) -> tuple[list[float], str]:
    """Ambil poligon (dinormalisasi 0-1). Multi-poligon digabung; fallback ke bbox."""
    seg = a.get("segmentation")
    polys = [p for p in seg if len(p) >= 6] if isinstance(seg, list) else []
    if polys:
        pts = [v for p in polys for v in p]
        kind = "multi_polygon" if len(polys) > 1 else "polygon"
    else:  # RLE / kosong -> pakai kotak bbox sebagai poligon 4 titik
        x, y, bw, bh = a["bbox"]
        pts = [x, y, x + bw, y, x + bw, y + bh, x, y + bh]
        kind = "bbox_fallback"
    norm = [min(max(v / (w if i % 2 == 0 else h), 0.0), 1.0) for i, v in enumerate(pts)]
    return norm, kind


def visualize(classes: list[str], n: int = 6) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    lbl_dir, img_dir = OUT / "labels" / "train", OUT / "images" / "train"
    files = [p for p in sorted(lbl_dir.glob("*.txt")) if p.stat().st_size > 0]
    for p in random.Random(SEED).sample(files, min(n, len(files))):
        img_path = next(img_dir.glob(p.stem + ".*"))
        img = Image.open(img_path)
        w, h = img.size
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis("off")
        for line in p.read_text(encoding="utf-8").splitlines():
            v = line.split()
            c, xy = int(v[0]), list(map(float, v[1:]))
            xs, ys = [x * w for x in xy[0::2]], [y * h for y in xy[1::2]]
            ax.fill(xs, ys, alpha=0.35)
            ax.plot(xs + [xs[0]], ys + [ys[0]], lw=1)
            ax.text(xs[0], ys[0], classes[c], fontsize=7, color="white",
                    backgroundcolor="black")
        fig.savefig(FIG / f"{p.stem}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    coco = json.loads(ANN.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    classes = split["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    site2split = {s: k for k in ("train", "val", "test") for s in split[k]}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    paths = {p.name: p for p in RAW.rglob("*") if p.suffix.lower() in IMG_EXT}

    # --- Pilih gambar per split + sampling background train ---
    rng = random.Random(SEED)
    selected = {"train": [], "val": [], "test": []}
    bg_train = []
    for im in coco["images"]:
        sp = site2split[im["site"]]
        if sp == "train" and not anns_by_img.get(im["id"]):
            bg_train.append(im)
        else:
            selected[sp].append(im)
    n_obj = len(selected["train"])
    n_bg = round(n_obj * BG_FRAC_TRAIN / (1 - BG_FRAC_TRAIN))
    selected["train"] += rng.sample(bg_train, min(n_bg, len(bg_train)))
    print(f"Background train: {len(bg_train):,} tersedia -> {min(n_bg, len(bg_train)):,} dipakai")

    # --- Tulis gambar & label ---
    if OUT.exists():
        shutil.rmtree(OUT)  # hapus hasil lama supaya tidak tercampur
    kinds, rows = Counter(), []
    for sp, ims in selected.items():
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for im in ims:
            name = Path(im["file_name"]).name
            shutil.copy2(paths[name], OUT / "images" / sp / name)
            lines = []
            for a in anns_by_img.get(im["id"], []):
                cls = split["class_map"][cat_name[a["category_id"]]]
                pts, kind = polygon_from_ann(a, im["width"], im["height"])
                kinds[kind] += 1
                lines.append(f"{cls_idx[cls]} " + " ".join(f"{v:.6f}" for v in pts))
            (OUT / "labels" / sp / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            rows.append({"file": name, "site": im["site"], "split": sp,
                         "n_instances": len(lines), "is_background": len(lines) == 0})

    # --- data.yaml ---
    yaml = (f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
            f"test: images/test\n\nnames:\n"
            + "".join(f'  {i}: "{c}"\n' for i, c in enumerate(classes)))
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")

    # --- Ringkasan ---
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    summary = man.groupby("split").agg(
        n_images=("file", "size"),
        n_background=("is_background", "sum"),
        n_instances=("n_instances", "sum"),
    ).loc[["train", "val", "test"]]
    OUT_M.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_M / "yolo_dataset_summary.csv")

    print("\nRINGKASAN DATASET YOLO")
    print(summary.to_string())
    print("\nJenis geometri anotasi:", dict(kinds))
    print(f"\ndata.yaml -> {OUT / 'data.yaml'}")

    visualize(classes)
    print(f"Gambar cek label -> {FIG}")


if __name__ == "__main__":
    main()