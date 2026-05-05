"""
01_textbias_identification.py  [optimized]
--------------------------------------------
GPU/CPU stage layout:
  §1–2  cudf   — large-scale string normalisation on raw 1.8M rows
  §3    pandas — NW alignment loop is row-by-row Python, GPU cannot help
  §4    pandas — street-type regex on aligned output (already filtered)
  §5    pandas — phonetic matching (rapidfuzz is CPU-only)
  §6    pandas — orthographic correction + re-alignment (row-by-row tqdm)
  §7    cudf   — 100-word frequency scan over not_matched_brute (GPU str.contains)

Exactly one cudf→pandas transfer (before §3) and one pandas→cudf
transfer (at the start of §7 for the relevant subset).

Usage:
    python 01_textbias_identification.py <geocoded_patients.csv> [OPTIONS]

Options:
    --gpu              CUDA device ID (default: 2)
    --output-dir       Output directory (default: ./data/outputs/)
    --odonymes         Path to odonymes.txt (default: ./data/odonymes.txt)
    --prenoms          Path to Prenoms.csv  (default: ./data/Prenoms.csv)
    --chunk-size       NW alignment chunk size (default: 1000)
    --min-bias-count   Minimum token frequency to be kept as a bias candidate
                       (default: 2 for production; use 1 on small test datasets)
"""

import argparse
import os
import sys

# ── Args BEFORE any GPU import so CUDA_VISIBLE_DEVICES is set first ──────────
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--gpu",            type=str, default="2")
parser.add_argument("--output-dir",     default="./data/outputs/")
parser.add_argument("--odonymes",       default="./data/odonymes.txt")
parser.add_argument("--prenoms",        default="./data/Prenoms.csv")
parser.add_argument("--chunk-size",     type=int, default=1000)
parser.add_argument("--min-bias-count", type=int, default=2,
                    help="Min token frequency to qualify as a bias (default: 2). "
                         "Use 1 on small test datasets such as sample.csv.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cudf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.methods import *

os.makedirs(args.output_dir, exist_ok=True)
date = datetime.today().strftime('%Y%m%d')
print(f"[01] start — date={date}  gpu={args.gpu}")

# ── Compile expensive regex patterns once ─────────────────────────────────────
_VOIE_TYPES = (
    r"AVENUE|RUE|PLACE|IMPASSE|BOULEVARD|ALLEE|SQUARE|ROUTE|ESPLANADE|CHEMIN|"
    r"GRANDE\sRUE|ROND\sPOINT|FAUBOURG|JARDIN|VIA|GALERIE|VOIE|QUAI|PASSAGE|"
    r"COUR|COURS|CITE|PARVIS|HAMEAU|VILLA|VILLAGE|VILLE|TERRASSE|PROMENADE|"
    r"SENTIER|CARREFOUR|CHAUSSEE|DOMAINE|CLOS|MOULIN|CENTRE|MAIL|BOIS|"
    r"PROMENEE|VALLEE|RESIDENCE|QUARTIER|LOTISSEMENT|TRAVERSE|LIEU\sDIT|"
    r"FERME|LE\sBOURG|PARC|COTE"
)
PAT_TYPO   = rf"\b(?:{_VOIE_TYPES})[A-Za-z]+"        # lane word merged with name (non-capturing)
PAT_SPLIT  = rf"(.*)({_VOIE_TYPES})(.*)"              # capture groups for split
PAT_VOIE_N = r"(?i)Voie [A-Za-z]+/\d+"               # BAN artefact


# ═════════════════════════════════════════════════════════════════════════════
# §1-2  CUDF — Load + normalise
# ═════════════════════════════════════════════════════════════════════════════
print("[01] §1 Loading data (cudf)...")

df_gpu      = cudf.read_csv(args.input_file, sep=";", dtype=str)
df_odonyme  = cudf.read_csv(args.odonymes, delimiter="|")
liste_odonyme_raw = cudf.Series(df_odonyme['synonym'].str.upper())

# liste_prenom is CPU-loaded; we clean it here as a cudf Series for §7 later
df_prenom   = pd.read_csv(args.prenoms, encoding="utf-8", sep=";")
df_prenom   = df_prenom.dropna(subset="01_prenom")
liste_prenom_raw = cudf.Series(df_prenom["01_prenom"])

print(f"  Loaded {len(df_gpu):,} rows")

print("[01] §2 Standardising (cudf)...")

df_gpu["street"] = df_gpu["result_housenumber"] + " " + df_gpu["result_name"]

mask_dom = (
    df_gpu["CPVILLE"].str.startswith("97") |
    df_gpu["CPVILLE"].str.startswith("98") |
    df_gpu["CPVILLE"].str.startswith("99")
)
df_gpu = df_gpu[~mask_dom].dropna(subset=["RUE", "street"])

df_gpu = df_gpu.rename(columns={
    "RUE":    "adr_init",
    "CPVILLE":"cp_init",
    "VILLE":  "ville_init",
    "street": "adr_geo",
    "result_postcode": "cp_geo",
    "result_city":     "ville_geo",
})

# normalize_address_series and remplacer_types_de_voies run on cudf → fast
df_gpu = normalize_address_series(df_gpu, "adr_init")
df_gpu = normalize_address_series(df_gpu, "adr_geo")
df_gpu = remplacer_types_de_voies(df_gpu, "adr_init", df_odonyme)
df_gpu = remplacer_types_de_voies(df_gpu, "adr_geo",  df_odonyme)

# Single normalize_spaces pass (two replaces = one is enough for terminal spaces)
for col in ("adr_init", "adr_geo"):
    df_gpu[col] = df_gpu[col].str.replace(r"\s+", " ", regex=True).str.strip()

print(f"  After filter: {len(df_gpu):,} rows")

# ── ONE transfer: cudf → pandas before the CPU-bound loop ────────────────────
df = df_gpu.to_pandas()
del df_gpu                   # free GPU memory early


# ═════════════════════════════════════════════════════════════════════════════
# §3  PANDAS — NW alignment  (row-by-row, GPU cannot help)
# ═════════════════════════════════════════════════════════════════════════════
print(f"[01] §3 NW alignment (pandas, chunk_size={args.chunk_size})...")

df = df.reset_index(drop=True)
chunks_out = []

for chunk_start in tqdm(range(0, len(df), args.chunk_size)):
    chunk = df.iloc[chunk_start : chunk_start + args.chunk_size].copy()

    for i in chunk.index:
        ad_brute = chunk.at[i, "adr_init"]
        ad_geo   = chunk.at[i, "adr_geo"]

        (char_score, token_score,
         brute_aligned, geo_aligned,
         unmatched_brute, unmatched_geo) = Needleman_Wunch_update(
            ad_brute, ad_geo, return_all=True
        )

        if pd.isna(char_score) or pd.isna(token_score):
            continue

        metrics = calculate_alignment_metrics(brute_aligned, geo_aligned)
        chunk.at[i, "score"]                 = metrics["alignment_score"]
        chunk.at[i, "char_alignment_score"]  = char_score
        chunk.at[i, "token_alignment_score"] = token_score
        chunk.at[i, "not_matched_brute"] = " ".join(unmatched_brute) if unmatched_brute else ""
        chunk.at[i, "not_matched_geo"]   = " ".join(unmatched_geo)   if unmatched_geo   else ""

    chunks_out.append(chunk)

# Single allocation instead of repeated concat inside the loop
df = pd.concat(chunks_out, ignore_index=True)
del chunks_out

# Checkpoint save — script 02 reads this file
out_aligned = os.path.join(args.output_dir, f"patients_geocoded_alignement_{date}.csv")
df.to_csv(out_aligned, sep=";", index=False)
print(f"  Aligned → {out_aligned}")


# ═════════════════════════════════════════════════════════════════════════════
# §4  PANDAS — Street-type position identification
# ═════════════════════════════════════════════════════════════════════════════
print("[01] §4 Street-type detection (pandas)...")

# odonyme as pandas Series for find_pos_elem in pandas mode
odonyme_pd = df_odonyme["terme"].to_pandas().str.upper().drop_duplicates()

df = find_pos_elem(df, "adr_init", odonyme_pd,
                   "street_type", "contains_street_type", "pos_street_type",
                   drop_col_elem=False)

# Catch lane-type tokens merged with street name (e.g. "RUEDE…")
no_street = df["contains_street_type"] == False
df.loc[no_street, "contains_street_type"] = (
    df.loc[no_street, "adr_init"].str.contains(PAT_TYPO, regex=True)
)

# Re-space those recovered rows
recoverable = df["contains_street_type"] & no_street
if recoverable.any():
    extracted = df.loc[recoverable, "adr_init"].str.extract(PAT_SPLIT, expand=True)
    extracted.columns = ["g1", "g2", "g3"]
    cleaned = (extracted["g1"] + " " + extracted["g2"] + " " + extracted["g3"]).str.strip()
    valid = cleaned.notna()
    df.loc[recoverable & valid, "adr_init"] = cleaned[valid]

    # Second pass only on fixed rows
    df = find_pos_elem(df, "adr_init", odonyme_pd,
                       "street_type", "contains_street_type", "pos_street_type",
                       drop_col_elem=False)

# Checkpoint — script 02 reads this
out_street = os.path.join(
    args.output_dir, f"patients_geocoded_aligned_{date}_wstreetInfos.csv"
)
df.to_csv(out_street, sep=";", index=False)
print(f"  Street infos → {out_street}")


# ═════════════════════════════════════════════════════════════════════════════
# §5  PANDAS — Geocoding-error detection (rapidfuzz is CPU-only)
# ═════════════════════════════════════════════════════════════════════════════
print("[01] §5 Geocoding-error detection (pandas)...")

# Remove BAN 'Voie XX/N' artefacts
df = df[~df["result_name"].str.contains(PAT_VOIE_N, regex=True, na=False)]

df["not_matched_brute"] = df["not_matched_brute"].fillna("")
df["not_matched_geo"]   = df["not_matched_geo"].fillna("")

mask_score         = df["score"].notna() & (df["score"] < 10)
mask_has_brute     = df["not_matched_brute"] != ""
mask_has_geo       = df["not_matched_geo"]   != ""
mask_low_char      = df["char_alignment_score"].notna() & (df["char_alignment_score"] < 85)
mask_name_mismatch = both_sides_have_name_tokens(df)

candidate_mask      = mask_has_brute & mask_has_geo & mask_low_char & mask_name_mismatch
mask_phonetic       = vectorized_is_phonetic_match(df, candidate_mask)
mask_filter2        = candidate_mask & ~mask_phonetic

df["geocoding_error"] = mask_score | mask_filter2

print(f"  Negative-score : {mask_score.sum():>7,}")
print(f"  Filter-2       : {mask_filter2.sum():>7,}")
print(f"  Total flagged  : {df['geocoding_error'].sum():>7,}")


# ═════════════════════════════════════════════════════════════════════════════
# §6  PANDAS — Ortho correction + re-alignment
# ═════════════════════════════════════════════════════════════════════════════
print("[01] §6 Orthographic correction & re-alignment (pandas)...")

# Drop addresses with fewer than 3 tokens that still carry a street type
short_mask = (
    df["adr_init"].str.split().str.len() < 3
) & (df["contains_street_type"] == True)

df_inc      = df[~short_mask]
df_clean    = df_inc[df_inc["geocoding_error"] == False]
df_flagged  = df_inc[df_inc["geocoding_error"] == True]

df_classed    = apply_unmatched_classification(df_clean)
df_corrected  = apply_ortho_correction_and_realign(df_classed)

df = pd.concat([df_corrected, df_flagged], ignore_index=True)

print(f"  Total rows             : {len(df):>7,}")
print(f"  Geocoding errors       : {df['geocoding_error'].sum():>7,}")
print(f"  With ortho corrections : {(df['ortho_pairs'] != '').sum():>7,}")

# Main checkpoint — scripts 03 and 04 read this
out_final = os.path.join(
    args.output_dir,
    f"patients_geocoded_alignement{date}_clean_classified.csv"
)
df.to_csv(out_final, sep=";", index=False)
print(f"  Classified → {out_final}")


# ═════════════════════════════════════════════════════════════════════════════
# §7  CUDF — Bias frequency scan  (100 × str.contains on GPU)
# ═════════════════════════════════════════════════════════════════════════════
print("[01] §7 Bias identification (cudf str.contains)...")

# Clean prenom / odonyme lists — cudf-specific string methods
for lst in (liste_prenom_raw, liste_odonyme_raw):
    lst = (lst.str.filter_alphanum(" ")
              .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
              .str.replace(r"\d", "", regex=True)
              .str.normalize_characters()
              .str.strip()
              .str.upper())

to_clean = cudf.concat([
    liste_prenom_raw, liste_odonyme_raw,
    cudf.Series(["BIS", "TER", "QUATER"]),
    cudf.Series(["LE", "LA", "LES", "A", "AUX", "DE", "DU", "DES", "L", "D", "E"]),
])

# Filter subsets in pandas (cheap), then convert to cudf for GPU str.contains
df_w_street_pd = df[
    (df["contains_street_type"] == True) & (df["geocoding_error"] == False)
]
to_clean_pd = to_clean.to_pandas().dropna().unique()
bruit_vrais = find_most_common_biases(df_w_street_pd, to_clean_pd)
bruit_vrais = bruit_vrais[bruit_vrais["count"] >= args.min_bias_count]
print(f"  Bias candidates (count >= {args.min_bias_count}): {len(bruit_vrais)}")

if len(bruit_vrais) == 0:
    print(
        f"\n  [WARNING] No bias tokens survived the count >= {args.min_bias_count} threshold.\n"
        f"  This is expected on small datasets (sample.csv has ~89 rows).\n"
        f"  Re-run with --min-bias-count 1 to lower the threshold:\n\n"
        f"    python scripts/01_textbias_identification.py {args.input_file} "
        f"--gpu {args.gpu} --min-bias-count 1\n\n"
        f"  Writing empty bias files and exiting."
    )
    pd.DataFrame(columns=["bias"]).to_csv("./data/biases_identified.csv", sep="|", index=False)
    raise SystemExit(0)

# Scan only the not_matched_brute column as a cudf Series — minimal transfer
has_unmatched   = df_w_street_pd["not_matched_brute"].notna() & (df_w_street_pd["not_matched_brute"] != "")
nmb_gpu         = cudf.Series(df_w_street_pd.loc[has_unmatched, "not_matched_brute"].values)

resultat = {
    mot: int(nmb_gpu.str.contains(rf"\b{mot}\b", regex=True).sum())
    for mot in bruit_vrais.head(100).index
}

freq_df = (
    pd.DataFrame(list(resultat.items()), columns=["Mot", "Occurrences"])
    .sort_values("Occurrences", ascending=False)
)
freq_df["freq_cumulee"] = freq_df["Occurrences"].cumsum()
freq_df["prop_cumul"]   = freq_df["freq_cumulee"] / freq_df["Occurrences"].sum()

out_biases_full = os.path.join(args.output_dir, f"biases_identified_{date}.csv")
freq_df.to_csv(out_biases_full, sep=";", index=False)
print(f"  Full bias list → {out_biases_full}")

# ── Compute dominant position per bias token ──────────────────────────────
# For each bias token, find whether it appears BEFORE or AFTER the lane type
# in the patient addresses. This position is saved alongside the token so
# script 03 can inject directly without re-reading the patient data.
top10_tokens = freq_df.head(10)["Mot"].tolist()

df_pos_analysis = df_w_street_pd[
    df_w_street_pd["contains_street_type"] == True
].copy()

odonyme_pd_pos = df_odonyme["terme"].to_pandas().str.upper().drop_duplicates()
df_pos_analysis = find_pos_elem(
    df_pos_analysis, "adr_init", odonyme_pd_pos,
    "street_type", "contains_street_type", "pos_street_type",
    drop_col_elem=False,
)

positions = {}
for tok in top10_tokens:
    mask = df_pos_analysis["not_matched_brute"].str.contains(
        rf"\b{tok}\b", regex=True, na=False
    )
    sub = df_pos_analysis[mask].copy()
    if len(sub) == 0:
        positions[tok] = "AFTER"   # default when no patient data to learn from
        continue
    sub["diff"] = sub["pos_street_type"] - sub["not_matched_brute"].str.split().apply(
        lambda tokens: float(tokens.index(tok)) if tok in tokens else np.nan
    )
    sub = sub.dropna(subset=["diff"])
    if len(sub) == 0:
        positions[tok] = "AFTER"
        continue
    before = (sub["diff"] > 0).sum()
    after  = (sub["diff"] < 0).sum()
    positions[tok] = "BEFORE" if before >= after else "AFTER"

top10 = pd.DataFrame([
    {"bias": tok, "position": positions[tok]}
    for tok in top10_tokens
])
top10.to_csv("./data/biases_identified.csv", sep="|", index=False)
print(f"  Top-10 biases with positions → ./data/biases_identified.csv")
print(top10.to_string(index=False))

print("[01] DONE.")
print(f"  → script 02 input : {out_street}")
print(f"  → script 03 & 04  : {out_final}")