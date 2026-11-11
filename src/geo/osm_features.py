"""
Phase 3 (geospasial) — Unduh fitur lingkungan dari OpenStreetMap untuk wilayah studi.
Lapisan: jalan, sungai/saluran, fasilitas sampah resmi, pasar.

Jalankan:  python -m src.geo.osm_features
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd


# Tampilkan progres di terminal & beri waktu tunggu lebih lama
ox.settings.log_console = True
# ox.settings.overpass_url = "https://overpass.kumi.systems/api"
ox.settings.requests_timeout = 300

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "data" / "raw" / "admin" / "study_area.gpkg"
OUT = ROOT / "data" / "raw" / "osm"
FIG = ROOT / "reports" / "figures"

METRIC_CRS = "EPSG:32748"
BUFFER_M = 1000  # ambil data 1 km di luar batas, supaya sel tepi tetap punya tetangga

LAYERS = {
    "roads": {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                          "unclassified", "residential", "service", "living_street"]},
    "waterways": {"waterway": ["river", "stream", "canal", "drain", "ditch"]},
    "waste_facilities": {"amenity": ["waste_transfer_station", "recycling", "waste_disposal"],
                         "landuse": ["landfill"]},
    "markets": {"amenity": ["marketplace"]},
}
LINE_LAYERS = {"roads", "waterways"}          # disimpan sebagai garis
KEEP = ["name", "highway", "waterway", "amenity", "landuse"]


def fetch(name: str, poly, tags: dict) -> gpd.GeoDataFrame | None:
    try:
        gdf = ox.features_from_polygon(poly, tags)
    except Exception as e:  # tidak ada data / Overpass sibuk
        print(f"  [{name}] gagal/kosong: {e}")
        return None
    gdf = gdf.reset_index()
    cols = [c for c in ["element", "id"] + KEEP if c in gdf.columns]
    gdf = gdf[cols + ["geometry"]]
    if name in LINE_LAYERS:
        gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
    else:  # titik & poligon -> titik representatif (untuk hitung jarak)
        gdf["geometry"] = gdf.geometry.representative_point()
    return gdf.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    area = gpd.read_file(AREA)
    poly = (area.to_crs(METRIC_CRS).geometry.union_all()
            .buffer(BUFFER_M))
    poly = gpd.GeoSeries([poly], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]

    print("Mengunduh dari OpenStreetMap (bisa 1-3 menit per lapisan)...")
    results = {} 
    for name, tags in LAYERS.items():
        path = OUT / f"{name}.gpkg"
        if path.exists():                       # sudah pernah diunduh -> baca saja
            gdf = gpd.read_file(path)
            print(f"  [{name}] sudah ada, dibaca dari file")
        else:
            gdf = fetch(name, poly, tags)
            if gdf is None or gdf.empty:continue
            gdf.to_file(path, driver="GPKG")
        results[name] = gdf
        if name in LINE_LAYERS:
            km = gdf.to_crs(METRIC_CRS).length.sum() / 1000
            print(f"  {name:17s}: {len(gdf):6,} garis | total {km:,.0f} km")
        else:
            print(f"  {name:17s}: {len(gdf):6,} titik")

    # Rincian fasilitas sampah (penting untuk definisi 'resmi' vs 'liar')
    if "waste_facilities" in results:
        wf = results["waste_facilities"]
        tipe = wf["amenity"].fillna(wf.get("landuse", pd.Series(index=wf.index)))
        print("\nRincian fasilitas sampah:")
        print(tipe.value_counts().to_string())

    # Peta cek visual
    fig, ax = plt.subplots(figsize=(10, 8))
    area.boundary.plot(ax=ax, color="black", linewidth=1.2)
    style = {"roads": ("lightgray", 0.3), "waterways": ("tab:blue", 0.8)}
    for name, (color, lw) in style.items():
        if name in results:
            results[name].plot(ax=ax, color=color, linewidth=lw)
    if "markets" in results:
        results["markets"].plot(ax=ax, color="orange", markersize=18, label="Pasar")
    if "waste_facilities" in results:
        results["waste_facilities"].plot(ax=ax, color="red", markersize=18,
                                         label="Fasilitas sampah (OSM)")
    ax.set_title("Fitur lingkungan OSM — Kota Bandung & Cimahi")
    ax.legend(loc="lower left")
    ax.set_axis_off()
    fig.savefig(FIG / "osm_features.png", dpi=150, bbox_inches="tight")
    print(f"\nPeta -> {FIG / 'osm_features.png'}")


if __name__ == "__main__":
    main()