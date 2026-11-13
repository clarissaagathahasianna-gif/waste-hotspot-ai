"""
Phase 6-7 — Tabel fitur + label per sel grid 500 m (dataset model prediksi hotspot).

Input:
- data/interim/grid_500m.gpkg            (grid analisis, Langkah 2)
- data/raw/admin/study_area.gpkg         (batas Kota Bandung & Cimahi)
- data/raw/osm/*.gpkg                    (jalan, sungai, pasar, fasilitas sampah)
- data/catalog/bandung_incidents.csv     (katalog insiden = sumber label)

Output:
- data/processed/grid_features_500m.csv / .gpkg
- reports/metrics/grid_features_by_label.csv
- reports/figures/grid_labels_500m.png

Jalankan:  python -m src.geo.build_grid_features
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data" / "interim" / "grid_500m.gpkg"
AREA = ROOT / "data" / "raw" / "admin" / "study_area.gpkg"
OSM = ROOT / "data" / "raw" / "osm"
CAT = ROOT / "data" / "catalog" / "bandung_incidents.csv"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MET = ROOT / "reports" / "metrics"

CRS = "EPSG:32748"
STUDY_CITIES = {"Kota Bandung", "Kota Cimahi"}
LABEL_PRECISION = {"exact", "street"}
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}
RIVERS = {"river", "stream", "canal"}  # drain/ditch tidak dipakai: pemetaannya di OSM tidak merata

FEATURES = ["dist_road_any_m", "dist_road_major_m", "road_density_km_km2",
            "dist_river_m", "dist_market_m", "dist_osm_waste_facility_m"]


def load_osm(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(OSM / f"{name}.gpkg").to_crs(CRS)


def nearest_distance(points: gpd.GeoDataFrame, target: gpd.GeoDataFrame) -> pd.Series:
    """Jarak (meter) dari titik pusat tiap sel ke fitur target terdekat."""
    j = gpd.sjoin_nearest(points[["cell_id", "geometry"]], target[["geometry"]],
                          how="left", distance_col="dist")
    return j.groupby("cell_id")["dist"].min()


def road_density(cells_in: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Panjang jalan (km) per km2 bagian sel yang berada di dalam wilayah studi."""
    pieces = gpd.overlay(roads[["geometry"]], cells_in[["cell_id", "geometry"]],
                         how="intersection", keep_geom_type=True)
    km = (pieces.length / 1000).groupby(pieces["cell_id"]).sum()
    area_km2 = cells_in.set_index("cell_id").geometry.area / 1e6
    return (km / area_km2).reindex(area_km2.index).fillna(0.0)


def load_incidents() -> gpd.GeoDataFrame:
    df = pd.read_csv(CAT, dtype=str, keep_default_na=False)
    keep = (df["city"].isin(STUDY_CITIES) & df["geocode_precision"].isin(LABEL_PRECISION)
            & (df["lat"] != "") & (df["needs_review"] == "no"))
    df = df[keep].copy()
    date_txt = df["date_event"].where(df["date_event"] != "", df["date_published"])
    df["date"] = pd.to_datetime(date_txt, errors="coerce")
    pts = gpd.points_from_xy(df["lon"].astype(float), df["lat"].astype(float))
    return gpd.GeoDataFrame(df, geometry=pts, crs="EPSG:4326").to_crs(CRS)


def label_cells(grid: gpd.GeoDataFrame, inc: gpd.GeoDataFrame) -> tuple[pd.DataFrame, int]:
    j = gpd.sjoin(inc, grid[["cell_id", "geometry"]], how="left", predicate="intersects")
    j = j.drop_duplicates("incident_id")          # titik tepat di batas sel -> ambil satu
    n_outside = int(j["cell_id"].isna().sum())
    j = j.dropna(subset=["cell_id"])
    agg = j.groupby("cell_id").agg(
        n_incidents=("incident_id", "size"),
        n_sites=("site_key", "nunique"),
        n_observed=("evidence_level", lambda s: int((s == "observed").sum())),
        n_illegal=("is_official_tps", lambda s: int((s != "yes").sum())),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    return agg, n_outside


def main() -> None:
    for p in (OUT, FIG, MET):
        p.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(GRID).to_crs(CRS)
    area = gpd.read_file(AREA).to_crs(CRS)
    union = area.geometry.union_all()
    print(f"Grid: {len(grid):,} sel")

    cent = grid[["cell_id"]].copy()
    cent = gpd.GeoDataFrame(cent, geometry=grid.centroid, crs=CRS)
    cells_in = grid[["cell_id"]].copy()
    cells_in = gpd.GeoDataFrame(cells_in, geometry=grid.intersection(union), crs=CRS)

    # --- Wilayah (Bandung/Cimahi) per sel ---
    reg = gpd.sjoin_nearest(cent, area[["region", "geometry"]], how="left")
    region = reg.drop_duplicates("cell_id").set_index("cell_id")["region"]

    # --- Fitur dari OSM ---
    print("Menghitung fitur (bisa 1-3 menit)...")
    roads = load_osm("roads")
    roads_major = roads[roads["highway"].astype(str).isin(MAJOR_ROADS)]
    water = load_osm("waterways")
    rivers = water[water["waterway"].astype(str).isin(RIVERS)]
    markets = load_osm("markets")
    waste = load_osm("waste_facilities")

    feat = pd.DataFrame(index=pd.Index(grid["cell_id"], name="cell_id"))
    feat["region"] = region
    feat["frac_inside"] = grid.set_index("cell_id")["frac_inside"]
    feat["x"] = cent.set_index("cell_id").geometry.x.round(1)
    feat["y"] = cent.set_index("cell_id").geometry.y.round(1)
    feat["dist_road_any_m"] = nearest_distance(cent, roads)
    feat["dist_road_major_m"] = nearest_distance(cent, roads_major)
    feat["road_density_km_km2"] = road_density(cells_in, roads)
    feat["dist_river_m"] = nearest_distance(cent, rivers)
    feat["dist_market_m"] = nearest_distance(cent, markets)
    feat["dist_osm_waste_facility_m"] = nearest_distance(cent, waste)
    feat[FEATURES] = feat[FEATURES].round(1)

    # --- Label dari katalog insiden ---
    inc = load_incidents()
    agg, n_outside = label_cells(grid, inc)
    feat = feat.join(agg)
    for c in ["n_incidents", "n_sites", "n_observed", "n_illegal"]:
        feat[c] = feat[c].fillna(0).astype(int)
    feat["label"] = (feat["n_incidents"] > 0).astype(int)
    feat["label_illegal"] = (feat["n_illegal"] > 0).astype(int)
    for c in ["first_date", "last_date"]:
        feat[c] = feat[c].dt.strftime("%Y-%m-%d")

    # --- Simpan ---
    table = feat.reset_index()
    table.to_csv(OUT / "grid_features_500m.csv", index=False)
    grid[["cell_id", "geometry"]].merge(table, on="cell_id").to_file(
        OUT / "grid_features_500m.gpkg", driver="GPKG")

    by_label = table.groupby("label")[FEATURES].median().round(1)
    by_label.to_csv(MET / "grid_features_by_label.csv")

    # --- Peta label ---
    gdf = grid[["cell_id", "geometry"]].merge(table[["cell_id", "label"]], on="cell_id")
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf[gdf["label"] == 0].plot(ax=ax, color="whitesmoke", edgecolor="lightgray", linewidth=0.3)
    gdf[gdf["label"] == 1].plot(ax=ax, color="tab:red", edgecolor="darkred", linewidth=0.5)
    area.boundary.plot(ax=ax, color="black", linewidth=1)
    inc.plot(ax=ax, color="black", markersize=10)
    ax.set_title("Sel grid 500 m dengan insiden sampah (merah) — Kota Bandung & Cimahi")
    ax.set_axis_off()
    fig.savefig(FIG / "grid_labels_500m.png", dpi=150, bbox_inches="tight")

    # --- Ringkasan ---
    n_pos = int(table["label"].sum())
    print("\nRINGKASAN DATASET")
    print(f"  Insiden dipakai sebagai label : {len(inc)} baris "
          f"({inc['site_key'].nunique()} lokasi unik); di luar grid: {n_outside}")
    print(f"  Sel positif (label=1)         : {n_pos} dari {len(table)} ({n_pos / len(table):.1%})")
    print(f"  Sel positif (hanya TPS liar)  : {int(table['label_illegal'].sum())}")
    print(f"  Nilai kosong per fitur        : {table[FEATURES].isna().sum().to_dict()}")
    print("\nMedian fitur: sel tanpa insiden (0) vs dengan insiden (1)")
    print(by_label.T.to_string())
    print(f"\nTabel  -> {OUT / 'grid_features_500m.csv'}")
    print(f"Peta   -> {FIG / 'grid_labels_500m.png'}")


if __name__ == "__main__":
    main()