"""
03_ref_biased_integration.py  [optimized]
-------------------------------------------
Reads the bias list + dominant positions saved by script 01,
injects each bias into a reference address dataset and geocodes
the variants in chunks via the BAN API.

No patient data is needed — all information comes from biases_identified.csv
which is the direct output of script 01.

Usage:
    python 03_ref_biased_integration.py [OPTIONS]

Options:
    --biases-file    Path to biases_identified.csv
                      (default: ./data/biases_identified.csv)
    --reference-file  Reference address CSV (auto-detected if omitted)
    --sample-size     Rows to sample from reference (default: 15000)
"""

import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--biases-file",   default="./data/biases_identified.csv")
parser.add_argument("--reference-file", default=None)
parser.add_argument("--sample-size",    type=int, default=15000)
args = parser.parse_args()

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.chunkcsv import process_dataframe_in_chunks2

date = datetime.today().strftime("%Y%m%d")
print(f"[03] start — date={date}")

DIRS = {
    "biased":   "./data/reference/ref_biased/",
    "chunked":   "./data/reference/ref_chunked/",
    "geo":       "./data/reference/ref_chunked_geocoded/",
    "out":       "./data/reference/ref_biased_geocoded/",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# §1  Load bias list with positions (direct output of script 01)
# ═════════════════════════════════════════════════════════════════════════════
print("[03] §1 Loading bias list...")

if not os.path.exists(args.biases_file):
    raise FileNotFoundError(
        f"{args.biases_file} not found. Run script 01 first."
    )

df_biases = pd.read_csv(args.biases_file, delimiter="|").dropna()

if len(df_biases) == 0:
    print("  [INFO] biases_identified.csv is empty — nothing to geocode.")
    raise SystemExit(0)

if "position" not in df_biases.columns:
    raise KeyError(
        "Column 'position' not found in biases_identified.csv.\n"
        "Re-run script 01 to regenerate the file with position information."
    )

# {bias_token: dominant_position} dict — built directly from the file
dict_pos_max = dict(zip(
    df_biases["bias"].astype(str).str.upper(),
    df_biases["position"].astype(str).str.upper(),
))

print(f"  {len(dict_pos_max)} biases loaded:")
for b, pos in dict_pos_max.items():
    print(f"    {b:15s} → {pos}")


# ═════════════════════════════════════════════════════════════════════════════
# §2  Load reference dataset and inject biases
# ═════════════════════════════════════════════════════════════════════════════
print("[03] §2 Loading reference dataset...")

if args.reference_file:
    ref_path = args.reference_file
else:
    ref_dir = "./data/reference/"
    candidates = [
        f for f in os.listdir(ref_dir)
        if f.endswith(".csv") and "etablissements" in f
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No reference CSV found in {ref_dir}. Pass --reference-file explicitly."
        )
    ref_path = os.path.join(ref_dir, candidates[0])
    print(f"  Auto-detected: {ref_path}")

df_ref = pd.read_csv(ref_path, sep=";", dtype=str)

mask_dom = (
    df_ref["code_postal_uai"].str.startswith("97") |
    df_ref["code_postal_uai"].str.startswith("98") |
    df_ref["code_postal_uai"].str.startswith("99")
)
df_ref_metrop = df_ref[~mask_dom].dropna(subset=["adresse_uai"]).copy()
df_ref_metrop["adresse_uai"] = df_ref_metrop["adresse_uai"].str.upper()

KEEP_COLS = [
    "numero_uai", "adresse_uai", "localite_acheminement_uai",
    "code_postal_uai", "coordonnee_x", "coordonnee_y", "latitude", "longitude",
]
df_sample = df_ref_metrop.sample(min(args.sample_size, len(df_ref_metrop)))[KEEP_COLS]
print(f"  Reference sample: {len(df_sample):,} rows")

print("[03] §2b Generating biased reference files...")

def inject_bias(df_src: pd.DataFrame, bias: str, pos: str, col: str) -> pd.DataFrame:
    out = df_src.copy()
    if pos == "AFTER":
        out[col] = out[col] + f" {bias}"
    elif pos == "BEFORE":
        out[col] = f"{bias} " + out[col]
    return out

for bias_token, position in dict_pos_max.items():
    out_path = os.path.join(DIRS["biased"], f"df_ref_{bias_token}.csv")
    inject_bias(df_sample, bias_token, position, "adresse_uai").to_csv(out_path)
    print(f"  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# §3  Geocode biased datasets in chunks (CPU/HTTP-bound)
# ═════════════════════════════════════════════════════════════════════════════
print("[03] §3 Geocoding biased datasets...")

GEOCODE_COLS = ["adresse_uai", "code_postal_uai", "localite_acheminement_uai"]
OUT_COLS = [
    "numero_uai", "adresse_uai", "localite_acheminement_uai", "code_postal_uai",
    "x_L93_ref", "y_L93_ref", "x_WGS84_ref", "y_WGS84_ref",
    "latitude", "longitude",
    "result_housenumber", "result_name", "result_postcode",
    "result_city", "result_label", "label",
]

t0 = time.time()
bias_files = [
    f for f in os.listdir(DIRS["biased"])
    if os.path.isfile(os.path.join(DIRS["biased"], f))
]

for fname in tqdm(bias_files):
    label = fname.split("_")[2].split(".")[0]

    df_to_geocode = (
        pd.read_csv(os.path.join(DIRS["biased"], fname), index_col=0)
        .rename(columns={
            "coordonnee_x": "x_L93_ref",
            "coordonnee_y": "y_L93_ref",
            "longitude":    "x_WGS84_ref",
            "latitude":     "y_WGS84_ref",
        })
    )
    df_to_geocode["label"] = label

    process_dataframe_in_chunks2(
        df=df_to_geocode,
        label=label,
        chunk_size=1000,
        chunks_dir_path=DIRS["chunked"],
        chunks_geo_dir_path=DIRS["geo"],
        columns={"columns": GEOCODE_COLS},
    )

    geocoded_files = [
        f for f in os.listdir(DIRS["geo"])
        if os.path.isfile(os.path.join(DIRS["geo"], f)) and f.startswith(f"{label}_")
    ]
    if not geocoded_files:
        print(f"  Warning: no geocoded chunks for '{label}' — skipping.")
        continue

    result = pd.concat(
        [pd.read_csv(os.path.join(DIRS["geo"], f)) for f in geocoded_files],
        ignore_index=True,
    )
    result = result[[c for c in OUT_COLS if c in result.columns]]
    result.to_csv(os.path.join(DIRS["out"], f"{label}_ref_biased.csv"), sep=";")
    print(f"  Geocoded → {DIRS['out']}{label}_ref_biased.csv")

elapsed = (time.time() - t0) / 60
print(f"[03] DONE in {elapsed:.1f} min.")