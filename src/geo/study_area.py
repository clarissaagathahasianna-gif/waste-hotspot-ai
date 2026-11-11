"""
Phase 1 — Menentukan & memvalidasi wilayah studi (Kota Bandung + Kota Cimahi)
dan membuat grid analisis.

Jalankan dari ROOT proyek:
    python -m src.geo.study_area
"""
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import box

# ---------- Konfigurasi ----------
ROOT = Path(__file__).resolve().parents[2]
RAW_ADMIN = ROOT / "data" / "raw" / "admin"
INTERIM = ROOT / "data" / "interim"
FIG = ROOT / "reports" / "figures"
for p in (RAW_ADMIN, INTERIM, FIG):
    p.mkdir(parents=True, exist_ok=True)

# Beberapa kandidat query per wilayah; dipilih yang luasnya masuk akal
REGIONS = {
    "Kota Bandung": ["Kota Bandung, Jawa Barat, Indonesia",
                     "Bandung, West Java, Indonesia"],
    "Kota Cimahi": ["Kota Cimahi, Jawa Barat, Indonesia",
                    "Cimahi, West Java, Indonesia"],
}
# Rentang sanity check luas (km2). Kota Bandung ±167 km2, Cimahi ±40 km2.
EXPECTED_KM2 = {"Kota Bandung": (150, 185), "Kota Cimahi": (35, 45)}

METRIC_CRS = "EPSG:32748"   # UTM zona 48S (meter) — cocok untuk Bandung
GEO_CRS = "EPSG:4326"
CELL_SIZES_M = [500, 250]   # 500 = utama, 250 = uji sensitivitas (MAUP)


def fetch_region(name: str, queries: list[str]) -> gpd.GeoDataFrame:
    """Coba tiap query; kembalikan yang luasnya sesuai ekspektasi."""
    lo, hi = EXPECTED_KM2[name]
    for q in queries:
        try:
            gdf = ox.geocode_to_gdf(q)
        except Exception as e:  # Nominatim tidak menemukan / timeout
            print(f"  [skip] '{q}': {e}")
            continue
        area_km2 = gdf.to_crs(METRIC_CRS).area.iloc[0] / 1e6
        print(f"  '{q}' → {gdf['display_name'].iloc[0][:70]}... | {area_km2:.1f} km2")
        if lo <= area_km2 <= hi:
            out = gdf[["geometry"]].copy()
            out["region"] = name
            out["area_km2"] = round(area_km2, 2)
            return out
    raise ValueError(f"Tidak ada query untuk {name} dengan luas {lo}-{hi} km2. "
                     "Cek manual di openstreetmap.org.")


def make_grid(area: gpd.GeoDataFrame, cell: int) -> gpd.GeoDataFrame:
    """Grid persegi (meter) yang beririsan dengan wilayah studi."""
    union = area.geometry.union_all()
    xmin, ymin, xmax, ymax = area.total_bounds
    xs = np.arange(xmin, xmax + cell, cell)
    ys = np.arange(ymin, ymax + cell, cell)
    cells = [box(x, y, x + cell, y + cell) for x in xs[:-1] for y in ys[:-1]]
    grid = gpd.GeoDataFrame(geometry=cells, crs=area.crs)
    grid = grid[grid.intersects(union)].reset_index(drop=True)
    grid["cell_id"] = [f"c{cell}_{i:05d}" for i in range(len(grid))]
    grid["frac_inside"] = (grid.intersection(union).area / cell**2).round(3)
    return grid


def main() -> None:
    print("1) Mengambil batas administrasi dari OpenStreetMap...")
    parts = [fetch_region(n, q) for n, q in REGIONS.items()]
    admin = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=GEO_CRS)
    admin.to_file(RAW_ADMIN / "study_area.gpkg", driver="GPKG")

    admin_m = admin.to_crs(METRIC_CRS)
    total_km2 = admin_m.area.sum() / 1e6
    print(f"\nTotal luas wilayah studi: {total_km2:.1f} km2")

    print("\n2) Membuat grid analisis...")
    for cell in CELL_SIZES_M:
        grid = make_grid(admin_m, cell)
        out = INTERIM / f"grid_{cell}m.gpkg"
        grid.to_file(out, driver="GPKG")
        print(f"  grid {cell} m: {len(grid):,} sel "
              f"({(grid.frac_inside < 0.5).sum()} sel tepi <50% di dalam) → {out.name}")

    print("\n3) Menyimpan peta cek visual...")
    m = folium.Map(location=[-6.91, 107.58], zoom_start=12, tiles="Esri.WorldStreetMap")
    folium.GeoJson(admin, tooltip=folium.GeoJsonTooltip(["region", "area_km2"])).add_to(m)
    grid500 = gpd.read_file(INTERIM / "grid_500m.gpkg").to_crs(GEO_CRS)
    folium.GeoJson(grid500, style_function=lambda _: {"weight": 0.3, "fillOpacity": 0}
                   ).add_to(m)
    m.save(FIG / "study_area.html")
    print(f"  → {FIG / 'study_area.html'} (buka di browser)")


if __name__ == "__main__":
    main()