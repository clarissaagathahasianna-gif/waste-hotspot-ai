"""
Phase 2 — Geocoding otomatis katalog insiden via Nominatim (OpenStreetMap).

Yang dilakukan script ini:
1. Membuat cadangan katalog (bandung_incidents.backup.csv)
2. Menyalin koordinat dari site_key yang SUDAH diverifikasi manual (needs_review == "no")
   ke baris lain dengan site_key yang sama
3. Mencari koordinat baris yang masih kosong lewat Nominatim, dibatasi ke kotak Bandung Raya
4. Menandai SEMUA hasil otomatis dengan needs_review = "yes"
5. Membuat peta review: reports/figures/catalog_review.html

Jalankan dari ROOT proyek:
    python -m src.geo.geocode_catalog
"""
import shutil
import time
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT / "data" / "catalog" / "bandung_incidents.csv"
BACKUP = ROOT / "data" / "catalog" / "bandung_incidents.backup.csv"
AREA = ROOT / "data" / "raw" / "admin" / "study_area.gpkg"
MAP = ROOT / "reports" / "figures" / "catalog_review.html"

NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "waste-hotspot-ai/0.1 (student portfolio project)"}
VIEWBOX = "107.25,-6.70,107.95,-7.25"  # lon_min,lat_max,lon_max,lat_min (Bandung Raya)
PAUSE_S = 1.1                           # kebijakan Nominatim: maks. 1 request/detik


def nominatim(query: str) -> dict | None:
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "id",
              "viewbox": VIEWBOX, "bounded": 1}
    r = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=30)
    time.sleep(PAUSE_S)
    r.raise_for_status()
    hits = r.json()
    return hits[0] if hits else None


def candidate_queries(row: pd.Series) -> list[str]:
    qs = []
    if row["geocode_query"].strip():
        qs.append(row["geocode_query"].strip())
    loc = row["location_text"].split("(")[0].split(",")[0].strip()
    qs.append(f"{loc}, {row['city']}")
    if loc.upper().startswith("TPS "):
        qs.append(f"{loc[4:]}, {row['city']}")
    return list(dict.fromkeys(qs))  # buang duplikat, urutan tetap


def copy_verified_sites(df: pd.DataFrame) -> int:
    verified = df[(df["lat"] != "") & (df["needs_review"] == "no")]
    site_xy = verified.groupby("site_key")[["lat", "lon"]].first()
    n = 0
    for i, row in df.iterrows():
        if row["lat"] == "" and row["site_key"] in site_xy.index:
            lat, lon = site_xy.loc[row["site_key"], ["lat", "lon"]]
            df.at[i, "lat"], df.at[i, "lon"] = lat, lon
            df.at[i, "geocode_source"] = f"disalin dari site_key {row['site_key']}"
            df.at[i, "needs_review"] = "no"
            n += 1
    return n


def geocode_missing(df: pd.DataFrame) -> tuple[int, int]:
    todo = df.index[df["lat"] == ""]
    print(f"\nMencari {len(todo)} lokasi di Nominatim (±{len(todo) * 2} detik)...")
    found = 0
    for i in todo:
        row = df.loc[i]
        hit, used = None, None
        for q in candidate_queries(row):
            try:
                hit = nominatim(q)
            except requests.RequestException as e:
                print(f"  [error] {row['incident_id']} '{q}': {e}")
                hit = None
            if hit:
                used = q
                break
        df.at[i, "needs_review"] = "yes"
        if hit:
            df.at[i, "lat"] = f"{float(hit['lat']):.6f}"
            df.at[i, "lon"] = f"{float(hit['lon']):.6f}"
            df.at[i, "geocode_source"] = f"nominatim: '{used}' -> {hit.get('display_name', '')[:90]}"
            found += 1
            print(f"  [OK] {row['incident_id']:7s} {used}")
        else:
            df.at[i, "geocode_source"] = "tidak ditemukan - isi manual"
            print(f"  [--] {row['incident_id']:7s} {row['location_text'][:60]}")
    return found, len(todo) - found


def review_map(df: pd.DataFrame) -> None:
    pts = df[df["lat"] != ""].copy()
    pts["lat_f"] = pts["lat"].astype(float)
    pts["lon_f"] = pts["lon"].astype(float)
    m = folium.Map(location=[-6.92, 107.60], zoom_start=11, tiles="Esri.WorldStreetMap")
    if AREA.exists():
        folium.GeoJson(gpd.read_file(AREA), name="Wilayah studi",
                       style_function=lambda _: {"fillOpacity": 0, "weight": 2, "color": "black"}
                       ).add_to(m)
    colors = {"no": "green", "yes": "red"}
    for _, r in pts.iterrows():
        html = (f"<b>{r['incident_id']}</b> ({r['geocode_precision']}, {r['evidence_level']})<br>"
                f"{r['location_text']}<br><i>{r['geocode_source']}</i>")
        folium.CircleMarker(
            [r["lat_f"], r["lon_f"]], radius=6, weight=1,
            color=colors.get(r["needs_review"], "gray"), fill=True, fill_opacity=0.8,
            popup=folium.Popup(html, max_width=350),
            tooltip=r["incident_id"],
        ).add_to(m)
    MAP.parent.mkdir(parents=True, exist_ok=True)
    m.save(MAP)


def main() -> None:
    df = pd.read_csv(CAT, dtype=str, keep_default_na=False)
    shutil.copy2(CAT, BACKUP)
    print(f"Katalog: {len(df)} baris | cadangan -> {BACKUP.name}")

    n_copy = copy_verified_sites(df)
    print(f"Koordinat disalin dari site_key terverifikasi: {n_copy} baris")

    found, missing = geocode_missing(df)
    df.to_csv(CAT, index=False)
    review_map(df)

    print("\nRINGKASAN")
    print(f"  Ditemukan otomatis : {found}")
    print(f"  Tidak ditemukan    : {missing}  (isi manual)")
    print(f"  Perlu dicek (merah): {(df['needs_review'] == 'yes').sum()}")
    print(f"\nPeta review -> {MAP}")


if __name__ == "__main__":
    main()