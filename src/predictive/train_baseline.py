"""
Phase 7 — Model baseline prediksi kerentanan hotspot (spatial block cross-validation).

Model yang dibandingkan:
  1. Acak            : patokan terendah (PR-AUC = proporsi sel positif)
  2. Aturan sederhana: skor = semakin dekat pasar, semakin rawan
  3. Logistic Regression (log-jarak, terstandardisasi, regularisasi L2)
  4. Random Forest
  5. LightGBM (jika terpasang; jika tidak -> HistGradientBoosting dari scikit-learn)

Validasi: Stratified Group K-Fold dengan grup = blok spasial 2 km x 2 km,
diulang beberapa kali dengan pembagian blok berbeda. Metrik dihitung dari prediksi
out-of-fold (gabungan semua fold), karena jumlah sel positif sangat sedikit.

Input : data/processed/grid_features_500m.csv
Output: reports/metrics/baseline_spatialcv.csv
        reports/metrics/lr_coefficients.csv
        reports/figures/baseline_risk_map_oof.png
        data/processed/oof_predictions_baseline.csv

Jalankan:  python -m src.predictive.train_baseline
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "grid_features_500m.csv"
MET = ROOT / "reports" / "metrics"
FIG = ROOT / "reports" / "figures"
OOF_OUT = ROOT / "data" / "processed" / "oof_predictions_baseline.csv"

FEATURES = ["dist_road_any_m", "dist_road_major_m", "road_density_km_km2",
            "dist_river_m", "dist_market_m", "dist_osm_waste_facility_m"]
TARGETS = ["label", "label_illegal"]   # utama & analisis sensitivitas (tanpa pasar/TPS resmi)
BLOCK_M = 2000                          # ukuran blok spasial
N_SPLITS = 5
N_REPEATS = 20
TOP_FRAC = 0.10                         # metrik "tangkapan" di 10% sel berisiko tertinggi


def make_models(seed: int) -> dict:
    lr = make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
    )
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if HAS_LGBM:
        gbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
                             random_state=seed, verbose=-1)
        gbm_name = "LightGBM"
    else:
        gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             min_samples_leaf=10, l2_regularization=1.0,
                                             class_weight="balanced", random_state=seed)
        gbm_name = "HistGradientBoosting"
    return {"Logistic Regression": lr, "Random Forest": rf, gbm_name: gbm}


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    k = max(1, int(round(TOP_FRAC * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    capture = y[top].sum() / y.sum()
    return {"ROC_AUC": roc_auc_score(y, score),
            "PR_AUC": average_precision_score(y, score),
            f"capture_top{int(TOP_FRAC * 100)}": capture}


def run_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    groups = df["block_id"].to_numpy()
    rng = np.random.default_rng(0)

    rows, oof_sum, model_names = [], {}, None
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=rep)
        models = make_models(seed=rep)
        model_names = list(models)
        oof = {name: np.zeros(len(y)) for name in models}
        for tr, te in cv.split(X, y, groups):
            for name, model in models.items():
                model.fit(X[tr], y[tr])
                oof[name][te] = model.predict_proba(X[te])[:, 1]
        # patokan tanpa model
        oof["Acak"] = rng.random(len(y))
        oof["Aturan: dekat pasar"] = -df["dist_market_m"].to_numpy()
        for name, score in oof.items():
            rows.append({"target": target, "repeat": rep, "model": name, **metrics(y, score)})
            if name in model_names:
                oof_sum[name] = oof_sum.get(name, 0) + score
    res = pd.DataFrame(rows)
    oof_mean = pd.DataFrame({f"{target}__{n}": s / N_REPEATS for n, s in oof_sum.items()})
    return res, oof_mean


def risk_map(df: pd.DataFrame, score: pd.Series, title: str, path: Path) -> None:
    col = ((df["x"] - df["x"].min()) / 500).round().astype(int)
    row = ((df["y"].max() - df["y"]) / 500).round().astype(int)
    img = np.full((row.max() + 1, col.max() + 1), np.nan)
    rank = score.rank(pct=True)
    img[row, col] = rank
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(img, cmap="YlOrRd", vmin=0, vmax=1)
    pos = df["label"] == 1
    ax.scatter(col[pos], row[pos], s=30, facecolors="none", edgecolors="black", linewidths=1.2,
               label="Sel dengan insiden")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Peringkat risiko (persentil)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    bx = np.floor(df["x"] / BLOCK_M).astype(int)
    by = np.floor(df["y"] / BLOCK_M).astype(int)
    df["block_id"] = bx.astype(str) + "_" + by.astype(str)

    print(f"Data: {len(df)} sel | blok spasial {BLOCK_M} m: {df['block_id'].nunique()} blok")
    print(f"Gradient boosting: {'LightGBM' if HAS_LGBM else 'HistGradientBoosting (LightGBM belum terpasang)'}")
    for t in TARGETS:
        print(f"  {t}: {int(df[t].sum())} sel positif ({df[t].mean():.1%})")
    print(f"\nMelatih {N_REPEATS} ulangan x {N_SPLITS} fold (bisa 1-3 menit)...")

    all_res, all_oof = [], []
    for t in TARGETS:
        res, oof = run_target(df, t)
        all_res.append(res)
        all_oof.append(oof)
    res = pd.concat(all_res)
    metric_cols = ["ROC_AUC", "PR_AUC", f"capture_top{int(TOP_FRAC * 100)}"]
    summary = (res.groupby(["target", "model"])[metric_cols]
               .agg(["mean", "std"]).round(3))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values(["target", "PR_AUC_mean"], ascending=[True, False])
    summary.to_csv(MET / "baseline_spatialcv.csv")

    oof = pd.concat([df[["cell_id", "x", "y", "label", "label_illegal"]]] + all_oof, axis=1)
    oof.to_csv(OOF_OUT, index=False)

    # Koefisien Logistic Regression (dilatih pada seluruh data, untuk interpretasi)
    lr = make_models(0)["Logistic Regression"].fit(df[FEATURES].to_numpy(float), df["label"])
    coef = pd.Series(lr[-1].coef_[0], index=FEATURES).sort_values()
    coef.round(3).to_frame("coef_std_log").to_csv(MET / "lr_coefficients.csv")

    best = summary.loc["label"].index[0]
    if best in ("Acak", "Aturan: dekat pasar"):
        best = summary.loc["label"].drop(["Acak", "Aturan: dekat pasar"]).index[0]
    risk_map(df, oof[f"label__{best}"],
             f"Peta risiko out-of-fold — {best} (spatial CV)", FIG / "baseline_risk_map_oof.png")

    pd.set_option("display.width", 160)
    print("\nHASIL SPATIAL CV (rata-rata ± std dari", N_REPEATS, "ulangan)")
    for t in TARGETS:
        print(f"\nTarget = {t}  (PR-AUC acak ≈ {df[t].mean():.3f})")
        print(summary.loc[t].to_string())
    print("\nKoefisien Logistic Regression (log-jarak terstandardisasi; negatif = makin dekat makin rawan):")
    print(coef.round(3).to_string())
    print(f"\nTabel -> {MET / 'baseline_spatialcv.csv'}")
    print(f"Peta  -> {FIG / 'baseline_risk_map_oof.png'}")


if __name__ == "__main__":
    main()