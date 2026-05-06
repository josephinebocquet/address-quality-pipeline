# Address Quality Pipeline for Large-Scale Patient Geocoding

A reproducible pipeline for identifying, classifying, and quantifying text biases in geocoded patient addresses at scale. Developed in the context of a French hospital system using a locally deployed instance of the [addok](https://github.com/addok/addok) geocoder.

---

## Context

When patient addresses are geocoded at scale, the geocoder silently drops information it cannot resolve: apartment numbers, building identifiers, institutional names, care facility labels. This creates a systematic gap between the raw address recorded in the hospital information system and the address that was actually used to produce a GPS coordinate.

This pipeline makes that gap measurable. It identifies which types of extra information are most common, locates addresses that are systematically misreferenced (hotspots), and quantifies the socio-economic error introduced when the geocoded location falls in a different IRIS zone than the true one.

---

## What the pipeline does

The pipeline runs in four sequential steps.

**Step 1 — Text bias identification** reads raw patient addresses and their geocoding results, aligns them using a Needleman-Wunsch sequence algorithm, and separates orthographic errors (typos resolved by the geocoder) from genuine extra information the geocoder ignored. The most frequent extra tokens are identified as the operational bias list, together with their dominant position relative to the street type (BEFORE or AFTER).

**Step 2 — Hotspot identification** detects addresses that appear an abnormally high number of times in the dataset — typically institutional or administrative addresses (care homes, hospitals) where many patients share the same official address. These are flagged for manual review before being excluded from bias analysis.

**Step 3 — Reference bias integration** reads the bias list produced by step 1 and injects each bias into a public reference address dataset (French school addresses) at the position identified in step 1. Each biased variant is then geocoded. The difference between the reference coordinate and the geocoded coordinate measures the spatial impact of each bias.

**Step 4 — Results analysis** joins geocoded points to IRIS zones, attaches median disposable income from the INSEE Filosofi dataset, and computes the income difference between the true IRIS zone and the geocoded IRIS zone. This quantifies the socio-economic error introduced by each type of text bias.

---

## Repository structure

```
address-quality-pipeline/
├── README.md
├── LICENSE.md               project licence + third-party attributions
├── environment.yml          conda environment (RAPIDS 24.12, Python 3.12)
├── sample.csv               anonymized example input (95 rows, all test cases)
│
├── notebooks/               exploratory reference notebooks (01–04)
├── scripts/
│   ├── run_pipeline.py      master entry point to run whole pipeline
│   ├── 01_textbias_identification.py
│   ├── 02_hotspot_identification.py
│   ├── 03_ref_biased_integration.py
│   └── 04_results_analysis.py
│
├── src/
│   ├── methods.py           shared library (GPU/CPU unified via is_cudf guards)
│   └── chunkcsv.py          chunked geocoding via HTTP
│
└── data/
    ├── odonymes.txt          lane-type synonym table (included)
    ├── Prenoms.csv           French given-name list (included)
    ├── biases_identified.csv      generated outputs by step 01 (gitignored) 
    ├── identified_admin_bias.csv   generated outputs by step 02 (gitignored)
    ├── socioeco/             included — provided with the repository
    │   ├── iris.geojson              IRIS boundaries (WGS84)
    │   └── BASE_TD_FILO_DISP_IRIS_2020.csv   INSEE Filosofi income data
    ├── reference/            included — provided with the repository
    │   └── fr-en-adresse-et-geolocalisation-etablissements-*.csv
    └── outputs/              generated outputs (gitignored)
```

Patient CSV data is never committed to this repository (CNIL compliance).

---

## Requirements

### Hardware

The pipeline runs on both GPU and CPU environments.

**GPU mode** (recommended for production on large datasets): step 01 uses RAPIDS cuDF for large-scale string operations. A CUDA-capable GPU is required. Steps 02, 03, and 04 run on CPU regardless.

**CPU-only mode**: if cudf is not available, step 01 detects this at startup and falls back to pandas automatically. All other steps are pandas-only and unaffected. Expect step 01 to be slower on large datasets but fully correct.

### Environment

```bash
conda env create -f environment.yml
conda activate adr_env
```

The `environment.yml` includes RAPIDS 24.12, Python 3.12, and all pip dependencies (`rapidfuzz`, `phonetic-fr`, `python-Levenshtein`, `fiona`, `seaborn`, etc.).

To install libpostal separately (optional — improves address normalisation):
```bash
conda install conda-forge::libpostal
```

---

## Geocoding setup (required for step 03)

Step 03 sends address variants to a geocoding server via HTTP. **This must be configured before running the pipeline.**

The geocoder is [addok](https://github.com/addok/addok), the engine behind the French national address API (BAN). For patient data, **a locally deployed instance is required** to ensure addresses never leave your infrastructure.

### Option A — Local addok instance via Docker (recommended for patient data)

```bash
# Pull and start the pre-indexed BAN image (~6 GB RAM, ~2 GB download)
docker run -d --name addok-ban -p 7878:7878 etalab/addok-france-bundle

# Verify
curl "http://localhost:7878/search/?q=10+avenue+emile+zola+paris&limit=1"

# Set the endpoint
export ADDOK_URL="http://localhost:7878/search/csv/"
```

For an instance hosted on a separate server (as in our deployment at HEGP):
```bash
export ADDOK_URL="http://<your-server-hostname>:7878/search/csv/"
```

### Option B — Public geocoding API (not suitable for patient data)

A public geocoding API is available at `https://data.geopf.fr/geocodage/search/csv/` but **must not be used for patient addresses** since requests leave your infrastructure. It may be used for testing with the anonymized `sample.csv` only.

```bash
export ADDOK_URL="https://data.geopf.fr/geocodage/search/csv/"
```

The `ADDOK_URL` environment variable is read by `src/chunkcsv.py` at runtime. Set it in your shell before running the pipeline, or add it to your conda environment activation script (`$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`).

---

## Data requirements

All reference files are included in the repository. No external downloads are needed before running.

| File | Location | Notes |
|---|---|---|
| Geocoded patient CSV | Not included (patient data) | Your input file |
| `odonymes.txt` | `data/odonymes.txt` | Included |
| `Prenoms.csv` | `data/Prenoms.csv` | Included |
| `iris.geojson` | `data/socioeco/` | Included |
| `BASE_TD_FILO_DISP_IRIS_2020.csv` | `data/socioeco/` | Included |
| School address reference CSV | `data/reference/` | Included |

---

## Quick start

```bash
# 1. Clone and set up
git clone <repository-url>
cd address-quality-pipeline
conda env create -f environment.yml
conda activate adr_env

# 2. Check available GPUs (optional — pipeline runs on CPU if none available)
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 3. Set geocoding endpoint (required for step 03)
export ADDOK_URL="http://<your-addok-server>:7878/search/csv/"

# 4. Run the full pipeline
# GPU mode (specify device):
python scripts/run_pipeline.py path/to/patients_geocoded.csv --gpu 2

# CPU-only mode (omit --gpu):
python scripts/run_pipeline.py path/to/patients_geocoded.csv

# 5. After step 02 completes, review hotspot candidates before step 03
#    Open data/identified_admin_bias.csv, remove false positives, save.

# 6. Resume from step 03
python scripts/run_pipeline.py path/to/patients_geocoded.csv --gpu 2 --steps 3,4
```

### Testing on the sample dataset

```bash
# GPU mode
python scripts/run_pipeline.py data/sample.csv --gpu 2 --min-bias-count 1

# CPU-only mode
python scripts/run_pipeline.py data/sample.csv --min-bias-count 1
```

`--min-bias-count 1` is needed for the sample (95 rows) because the default threshold of 2 requires tokens to appear at least twice — too strict for a small test dataset. On production data the default is appropriate and `--min-bias-count` should not be specified.

---

## License

The pipeline code and reference data files (`odonymes.txt`, `Prenoms.csv`, `sample.csv`, `iris.geojson`, `BASE_TD_FILO_DISP_IRIS_2020.csv`, school address CSV) are the authors' own work or redistributed under their respective open licences. See `LICENSE.md` for details.

**Third-party attributions** (licences apply to their respective components only):

| Component | Role | Licence |
|---|---|---|
| [addok](https://github.com/addok/addok) | Geocoding engine | MIT |
| [API Adresse / BAN](https://adresse.data.gouv.fr) | National address reference dataset | Etalab Open Licence 2.0 |
| [INSEE Filosofi](https://www.insee.fr/fr/statistiques/6036907) | IRIS median income data | Etalab Open Licence 2.0 |
| [IGN CONTOURS-IRIS](https://geoservices.ign.fr/irisge) | IRIS boundary polygons | Etalab Open Licence 2.0 |

Patient data is never included in this repository (CNIL compliance).