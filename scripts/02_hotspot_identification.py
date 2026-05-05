"""
02_hotspot_identification.py  
------------------------------------------
Usage:
    python 02_hotspot_identification.py <*_wstreetInfos.csv> [OPTIONS]

Options:
    --output-dir    Output directory (default: ./data/outputs/)
    --hospital-lat  Hospital latitude  WGS84 (default: 48.83931, HEGP)
    --hospital-lon  Hospital longitude WGS84 (default:  2.27537, HEGP)
    --min-count     Min occurrences to flag as hotspot (default: 30)
    --exclude-cp    Postal code to exclude from histogram (default: 75015)
"""
import argparse
import os
import sys
from datetime import datetime
 
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--output-dir",   default="./data/outputs/")
parser.add_argument("--hospital-lat", type=float, default=48.83931098867617)
parser.add_argument("--hospital-lon", type=float, default=2.2753718727661623)
parser.add_argument("--min-count",    type=int,   default=30)
parser.add_argument("--exclude-cp",   type=int,   default=75015)
args = parser.parse_args()
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.methods import calculer_distance_euclidienne
 
os.makedirs(args.output_dir, exist_ok=True)
 
 
# ── Column-name repair guard ──────────────────────────────────────────────────
# Defensive utility for an upstream pipeline behaviour that can silently replace
# column names with integer indices (e.g. after certain cudf read_csv / concat
# operations where the schema is not written to the file).
#
# Usage: call repair_column_names(df, reference_df) right after any read_csv
# or concat that exhibits this symptom, before passing the frame downstream.
#
#   df_loaded = pd.read_csv(path, sep=";")
#   if df_loaded.columns.dtype == int:          # symptom check
#       df_loaded = repair_column_names(df_loaded, reference_df)
 
def repair_column_names(df: pd.DataFrame,
                        reference: pd.DataFrame) -> pd.DataFrame:
    """
    Restore string column names when they have been replaced by integer indices.
 
    Parameters
    ----------
    df        : DataFrame with integer column indices (the broken frame).
    reference : Any DataFrame that has the correct column names in the same
                positional order (e.g. a small head() of the same file read
                with explicit dtype=str, or the source DataFrame before saving).
 
    Returns
    -------
    DataFrame with columns renamed to match *reference*.
    """
    expected = list(reference.columns)
    if len(df.columns) != len(expected):
        raise ValueError(
            f"Cannot repair: df has {len(df.columns)} columns, "
            f"reference has {len(expected)}."
        )
    return df.rename(columns=dict(zip(df.columns, expected)))
fig_dir = "./figures/"
os.makedirs(fig_dir, exist_ok=True)
 
date = datetime.today().strftime("%Y%m%d")
print(f"[02] start — date={date}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §1  Load — only the columns we actually use
# ═════════════════════════════════════════════════════════════════════════════
print("[02] §1 Loading data...")
 
df = pd.read_csv(args.input_file, sep=";", usecols=[
    "adr_init", "ville_init", "cp_init",
    "latitude", "longitude",
    "contains_street_type",
])
df["contains_street_type"] = df["contains_street_type"].astype(bool)
 
df_w_street = df[df["contains_street_type"]].copy()
print(f"  Total: {len(df):,}  |  With street type: {len(df_w_street):,}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §2  Group-count repeated addresses
# ═════════════════════════════════════════════════════════════════════════════
print("[02] §2 Computing group counts...")
 
GROUP_KEYS = ["adr_init", "ville_init", "cp_init"]
 
df_counts = (
    df_w_street
    .groupby(GROUP_KEYS)
    .agg(
        count=("adr_init", "size"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    .reset_index()
)
 
# agg already deduplicates — no need for a separate merge + drop_duplicates
print(f"  Distinct candidate addresses: {len(df_counts):,}")
print(f"  Max count: {df_counts['count'].max()}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §3  Distribution plots
# ═════════════════════════════════════════════════════════════════════════════
print("[02] §3 Saving plots...")
 
df_plot = df_counts[
    (df_counts["cp_init"] != args.exclude_cp) &
    (df_counts["count"]   >= args.min_count)
]
 
if len(df_plot) > 0:
    # Linear histogram
    fig, ax = plt.subplots()
    ax.hist(df_plot["count"], bins=min(50, len(df_plot)), edgecolor="black")
    ax.set(title="Distribution of grouped address counts",
           xlabel="Count", ylabel="Frequency")
    fig.savefig(os.path.join(fig_dir, f"count_distribution_{date}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Linear histogram saved → {fig_dir}")
 
    # Log-log histogram — only meaningful when there is a range of counts
    count_min = max(df_plot["count"].min(), 1)
    count_max = df_plot["count"].max()
    if count_min < count_max and len(df_plot) >= 5:
        bins = np.logspace(np.log10(count_min), np.log10(count_max), 50)
        fig, ax = plt.subplots()
        ax.hist(df_plot["count"], bins=bins, edgecolor="black")
        ax.set(xscale="log", yscale="log",
               title="Log-Log histogram",
               xlabel="Count (log)", ylabel="Frequency (log)")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        fig.savefig(os.path.join(fig_dir, f"count_loglog_{date}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Log-log histogram saved → {fig_dir}")
    else:
        print(f"  Log-log histogram skipped "
              f"(need ≥5 candidates with varied counts, got {len(df_plot)}).")
else:
    print("  No candidates above threshold — plots skipped.")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §4  GeoDataFrame + distance from hospital
# ═════════════════════════════════════════════════════════════════════════════
print("[02] §4 Computing distances (Lambert-93)...")
 
gdf = gpd.GeoDataFrame(
    df_counts,
    geometry=gpd.points_from_xy(df_counts["longitude"], df_counts["latitude"]),
    crs="EPSG:4326",
).to_crs("EPSG:2154")
 
# Hospital reference point
hosp = (
    gpd.GeoDataFrame(
        {"geometry": [Point(args.hospital_lon, args.hospital_lat)]},
        crs="EPSG:4326",
    )
    .to_crs("EPSG:2154")
)
 
gdf["distance_m"]  = calculer_distance_euclidienne(
    gdf.geometry.x.values,
    gdf.geometry.y.values,
    hosp.geometry.x.values,
    hosp.geometry.y.values,
)
gdf["distance_km"] = gdf["distance_m"] / 1000
 
 
# ═════════════════════════════════════════════════════════════════════════════
# §5  Flag hotspots and save
# ═════════════════════════════════════════════════════════════════════════════
print("[02] §5 Flagging hotspots...")
 
gdf["is_hotspot"] = gdf["count"] >= args.min_count
print(f"  Candidates flagged (count ≥ {args.min_count}): {gdf['is_hotspot'].sum():,}")
 
# ── Save candidate list for manual review ─────────────────────────────────
# The original notebook saves this file, then the analyst opens it, removes
# any false positives (e.g. a legitimately high-density residential address
# that is not an administrative bias), and saves it back before script 04
# reads it. The automated count-threshold above is a starting point, not a
# final decision. Review the file before running script 04.
out_hotspot = "./data/identified_admin_bias.csv"
gdf.drop(columns="geometry").to_csv(out_hotspot, sep=";", index=False)
print(f"  Candidate hotspots → {out_hotspot}")
print("  ⚠  Review this file and remove any false positives before running script 04.")
 
# Also write the patient-level merge (all rows with is_hotspot flag) for
# downstream use without re-running the groupby.
df_out = df.merge(
    gdf[GROUP_KEYS + ["count", "distance_km", "is_hotspot"]],
    on=GROUP_KEYS,
    how="left",
)
df_out["is_hotspot"] = df_out["is_hotspot"].infer_objects(copy=False).fillna(False)
 
out_full = os.path.join(args.output_dir, f"patients_geocoded_hotspot_{date}.csv")
df_out.to_csv(out_full, sep=";", index=False)
print(f"  Full patient-level → {out_full}")
print("[02] DONE.")