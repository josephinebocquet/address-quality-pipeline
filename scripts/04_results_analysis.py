"""
04_results_analysis.py  
-------------------------------------------------
Produces summary statistics, address-quality metrics, and bias impact plots.

Reference data expected under ./data/socioeco/:
  - iris.geojson                        IRIS boundaries in WGS84
                                         → generate once with build_iris_geojson.py
  - BASE_TD_FILO_DISP_IRIS_2020.csv     INSEE Filosofi income data

Also used:
  - ./data/identified_admin_bias.csv    Hotspot candidates from script 02
                                         (manually reviewed before this step)
  - ./data/reference/ref_biaised_geocoded/   Per-bias geocoded CSVs from script 03

Usage:
    python 04_results_analysis.py <*_clean_classified.csv> [OPTIONS]

Options:
    --hotspot-file   Reviewed hotspot CSV from script 02
                     (default: ./data/identified_admin_bias.csv)
    --biaises-file   Path to biaises_identified.csv
    --iris-file      IRIS GeoJSON (default: ./data/socioeco/iris.geojson)
    --revenus-file   Filosofi income CSV
                     (default: ./data/socioeco/BASE_TD_FILO_DISP_IRIS_2020.csv)
    --geocoded-dir   Per-bias geocoded CSVs from script 03
    --figures-dir    Output directory for plots
    --output-dir     Output directory for CSV results
"""

import argparse
import os
import sys
from datetime import datetime
 
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--hotspot-file",  default="./data/identified_admin_bias.csv")
parser.add_argument("--biaises-file",  default="./data/biaises_identified.csv")
parser.add_argument("--iris-file",     default="./data/socioeco/iris.geojson")
parser.add_argument("--revenus-file",
                    default="./data/socioeco/BASE_TD_FILO_DISP_IRIS_2020.csv")
parser.add_argument("--geocoded-dir",  default="./data/reference/ref_biaised_geocoded/")
parser.add_argument("--figures-dir",   default="./figures/")
parser.add_argument("--output-dir",    default="./data/outputs/")
args = parser.parse_args()
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.methods import find_pos_elem, freq_couverture, calculer_distance_euclidienne
 
os.makedirs(args.figures_dir, exist_ok=True)
os.makedirs(args.output_dir,  exist_ok=True)
 
date = datetime.today().strftime("%Y%m%d")
print(f"[04] start — date={date}")
 
 
# ── Column-name repair helper ─────────────────────────────────────────────────
# geopandas occasionally replaces column names with integer indices when
# constructing a GeoDataFrame from a DataFrame that went through the cudf proxy
# layer. This function restores the correct names from a reference DataFrame.
 
def repair_gdf_columns_from_df(df: pd.DataFrame,
                                gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Restore GeoDataFrame column names to match a reference DataFrame."""
    non_geo = [c for c in gdf.columns if c != "geometry"]
    if non_geo and isinstance(non_geo[0], int):
        gdf = gdf.rename(columns={i: name for i, name in enumerate(df.columns)})
    return gdf
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §1  Load
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §1 Loading data...")
 
df = pd.read_csv(args.input_file, sep=";").drop(columns=["Unnamed: 0"], errors="ignore")
df_biaises = pd.read_csv(args.biaises_file, delimiter="|").dropna()
biaises = pd.Series(df_biaises["biais"].astype(str).to_list())
print(f"  Rows: {len(df):,}  |  Biases: {len(biaises)}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §2  Bias position analysis
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §2 Bias position analysis...")
 
df_positions = find_pos_elem(
    df.copy(), "adr_init", biaises,
    "biais", "contains_biais", "pos_biais",
    drop_col_elem=False,
)
df_positions["comp_biais_street"] = ""  # object dtype — avoids float64 conflict on string assignment
 
mask_residual = (
    df_positions["contains_street_type"] &
    df_positions["contains_biais"] &
    df_positions["not_matched_brute"].notna()
)
diff = (
    df_positions.loc[mask_residual, "pos_street_type"]
    - df_positions.loc[mask_residual, "pos_biais"]
)
df_positions.loc[mask_residual, "comp_biais_street"] = np.where(
    diff > 0, "BEFORE", "AFTER"
)
mask_no_street = (
    ~df_positions["contains_street_type"] &
    df_positions["contains_biais"] &
    df_positions["not_matched_brute"].notna()
)
df_positions.loc[mask_no_street, "comp_biais_street"] = "NO STREET"
 
df_biaised = df_positions[df_positions["contains_biais"]].copy()
 
counts = df_biaised.groupby(["biais", "comp_biais_street"]).size().reset_index(name="n")
totals = counts.groupby("biais")["n"].sum().rename("total")
counts  = counts.join(totals, on="biais")
counts["pct"] = np.round(100 * counts["n"] / counts["total"], 2)
pivot = counts.pivot(index="biais", columns="comp_biais_street", values="pct")
 
df_biaised["id"] = range(len(df_biaised))
df_freq = (
    freq_couverture(df_biaised, biaises.to_list(), "adr_init")
    .sort_values("cumsum", ascending=False)[["cumsum", "freq_biais_cum"]]
)
summary = pd.concat([df_freq, pivot], axis=1)
 
out_summary = os.path.join(args.output_dir, f"bias_position_summary_{date}.csv")
summary.to_csv(out_summary, sep=";")
print(f"  → {out_summary}")
print(summary.to_string())
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §3  Address-quality breakdown
# ═════════════════════════════════════════════════════════════════════════════
print("\n[04] §3 Address-quality breakdown...")
 
hotspot_file = args.hotspot_file
if not os.path.exists(hotspot_file):
    print(f"  Hotspot file not found at {hotspot_file} — skipping hotspot breakdown.")
    print("  Run script 02 and review ./data/identified_admin_bias.csv first.")
    hotspot_file = None
 
total = len(df_positions)
if hotspot_file and os.path.exists(hotspot_file):
    df_hotspot = pd.read_csv(hotspot_file, sep=";")
    df_merged  = df_positions.merge(
        df_hotspot[["adr_init", "ville_init", "cp_init", "is_hotspot"]],
        on=["adr_init", "ville_init", "cp_init"], how="left",
    )
    df_merged["is_hotspot"] = df_merged["is_hotspot"].fillna(False)
    clean = df_merged[
        df_merged["contains_street_type"] &
        ~df_merged["contains_biais"] &
        ~df_merged["is_hotspot"]
    ]
    print(f"  Clean       : {len(clean):>7,}  ({100*len(clean)/total:.3f}%)")
    print(f"  Problematic : {total-len(clean):>7,}  ({100*(total-len(clean))/total:.3f}%)")
else:
    print("  Hotspot file not found — skipping.")
 
df_w  = df_positions[ df_positions["contains_street_type"]]
df_wo = df_positions[~df_positions["contains_street_type"]]
df_wb = df_positions[ df_positions["contains_biais"]]
print(f"\n  With street type    : {len(df_w):>7,}  ({100*len(df_w)/total:.3f}%)")
print(f"  Without street type : {len(df_wo):>7,}  ({100*len(df_wo)/total:.3f}%)")
print(f"  With known bias     : {len(df_wb):>7,}  ({100*len(df_wb)/total:.3f}%)")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §4  Load per-bias geocoded results
# ═════════════════════════════════════════════════════════════════════════════
print("\n[04] §4 Loading geocoded bias results...")
 
if not os.path.isdir(args.geocoded_dir):
    print("  Geocoded dir not found — skipping spatial analysis.")
    print("[04] DONE.")
    raise SystemExit(0)
 
geo_files = [
    os.path.join(args.geocoded_dir, f)
    for f in os.listdir(args.geocoded_dir) if f.endswith(".csv")
]
if not geo_files:
    print("  No geocoded files — skipping spatial analysis.")
    print("[04] DONE.")
    raise SystemExit(0)
 
df = pd.concat(
    [pd.read_csv(f, sep=";", index_col=0) for f in geo_files],
    ignore_index=True,
)
print(f"  {len(df):,} geocoded rows across {len(geo_files)} bias files.")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §5  IRIS + revenus
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §5 Loading IRIS and income data...")
 
if not os.path.exists(args.iris_file):
    raise FileNotFoundError(
        f"{args.iris_file} not found.\n"
    )
df_iris = gpd.read_file(args.iris_file)[["CODE_IRIS", "geometry"]]
print(f"  {len(df_iris):,} IRIS zones.")
 
if not os.path.exists(args.revenus_file):
    raise FileNotFoundError(
        f"{args.revenus_file} not found.\n"
        "Download BASE_TD_FILO_DISP_IRIS_2020.csv from:\n"
        "  https://www.insee.fr/fr/statistiques/6036907\n"
        "and place it in ./data/socioeco/"
    )
 
# "ns" = non significatif, "nd" = non disponible — both treated as NaN
df_revenus = (
    pd.read_csv(args.revenus_file, sep=";", dtype=str)[["IRIS", "DISP_MED20"]]
    .replace({"ns": np.nan, "nd": np.nan})
    .rename(columns={"IRIS": "CODE_IRIS"})
    .dropna(subset="CODE_IRIS")
    .drop_duplicates(subset="CODE_IRIS")
)
df_revenus["DISP_MED20"] = pd.to_numeric(df_revenus["DISP_MED20"], errors="coerce")
print(f"  {len(df_revenus):,} IRIS zones with income data.")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §6  Spatial join
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §6 Spatial join...")
 
def sjoin_iris(df_src: pd.DataFrame, lon_col: str, lat_col: str,
               df_iris: gpd.GeoDataFrame, result_col: str) -> pd.Series:
    gdf = gpd.GeoDataFrame(
        df_src[[lon_col, lat_col]],
        geometry=gpd.points_from_xy(df_src[lon_col], df_src[lat_col]),
        crs="EPSG:4326",
    )
    # Guard against the integer-column-index bug
    gdf = repair_gdf_columns_from_df(df_src[[lon_col, lat_col]], gdf)
    joined = gpd.sjoin(gdf, df_iris[["CODE_IRIS", "geometry"]],
                       how="left", predicate="within")
    return joined["CODE_IRIS"].rename(result_col)
 
df["CODE_IRIS_init"]     = sjoin_iris(df, "x_WGS84_ref", "y_WGS84_ref",
                                       df_iris, "CODE_IRIS_init")
df["CODE_IRIS_geocoded"] = sjoin_iris(df, "longitude",   "latitude",
                                       df_iris, "CODE_IRIS_geocoded")
 
df = df.dropna(subset=["CODE_IRIS_geocoded", "CODE_IRIS_init"])
 
df = (
    df
    .merge(df_revenus[["CODE_IRIS", "DISP_MED20"]], left_on="CODE_IRIS_geocoded",
           right_on="CODE_IRIS", how="left")
    .rename(columns={"DISP_MED20": "DISP_MED20_geocoded"})
    .drop(columns="CODE_IRIS", errors="ignore")
    .merge(df_revenus[["CODE_IRIS", "DISP_MED20"]], left_on="CODE_IRIS_init",
           right_on="CODE_IRIS", how="left")
    .rename(columns={"DISP_MED20": "DISP_MED20_init"})
    .drop(columns="CODE_IRIS", errors="ignore")
)
 
for col in ("DISP_MED20_geocoded", "DISP_MED20_init",
            "y_WGS84_ref", "x_WGS84_ref", "latitude", "longitude"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
df["distance_km"] = calculer_distance_euclidienne(
    df["y_WGS84_ref"].values, df["x_WGS84_ref"].values,
    df["latitude"].values,    df["longitude"].values,
)
df["distance_m"]  = df["distance_km"] * 1000
df["diff_revenu"] = df["DISP_MED20_init"] - df["DISP_MED20_geocoded"]
df["label"]       = df["label"].str.title()
 
print(f"  Rows after join : {len(df):,}")
print(f"  With income data: {df['DISP_MED20_init'].notna().sum():,}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §7  Plots
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §7 Generating plots...")
 
def styled_boxplot(data, y, ylabel, title, ylim, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x="label", y=y, data=data,
                whis=1.5, showcaps=True, showfliers=False, ax=ax)
    sns.stripplot(x="label", y=y, data=data,
                  jitter=True, color="black", alpha=0.3, size=2, ax=ax)
    for patch in ax.patches:
        patch.set_edgecolor("red"); patch.set_linewidth(2); patch.set_facecolor("none")
    ax.set(xlabel="Bias", ylabel=ylabel, title=title, ylim=ylim)
    plt.xticks(rotation=45); plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out_path}")
 
styled_boxplot(df, "diff_revenu", "Difference of income (€)",
               "Income difference per bias", (0, 1000),
               os.path.join(args.figures_dir, "boxplot_diff_revenu_zoom_1000e.png"))
 
styled_boxplot(df, "distance_km", "Distance (km)",
               "Geocoding error distance per bias", (0, 0.05),
               os.path.join(args.figures_dir, "boxplot_distance_km_zoom_0_50m.png"))
 
g = sns.FacetGrid(df, row="label", hue="label", aspect=5, height=1.2,
                  palette="muted", sharex=True, sharey=False)
g.map(sns.kdeplot, "distance_km", fill=True, alpha=0.8, linewidth=1.2)
g.map(plt.axhline, y=0, lw=1, clip_on=False)
g.set_titles(row_template="{row_name}")
g.set(yticks=[], ylabel=None)
plt.xlim(0, 1); g.set_xlabels("Distance from reference (km)")
plt.suptitle("Ridge plot of distances per bias")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_ridge = os.path.join(args.figures_dir, "ridgeplot_distance_km.png")
g.savefig(out_ridge); plt.close("all")
print(f"  → {out_ridge}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §8  Summary statistics
# ═════════════════════════════════════════════════════════════════════════════
print("[04] §8 Summary statistics...")
 
df_stats = (
    df.groupby("label")[["distance_km", "diff_revenu"]]
    .agg(["mean", "median"])
    .round(3)
)
df_stats.columns = ["dist_mean_km", "dist_med_km", "revenu_mean", "revenu_med"]
# fix: rename needs columns= kwarg, not a positional dict
df_stats = df_stats.reset_index().rename(columns={"label": "biais"})
 
out_stats = os.path.join(args.output_dir, f"bias_impact_stats_{date}.csv")
df_stats.to_csv(out_stats, sep=";", index=False)
print(f"  → {out_stats}")
print(df_stats.to_string(index=False))
 
out_html = os.path.join(args.output_dir, f"bias_impact_stats_{date}.html")
df_stats.to_html(out_html, index=False, border=1)
print(f"  → {out_html}")
 
print("[04] DONE.")