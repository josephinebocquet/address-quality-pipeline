"""
run_pipeline.py
----------------
Master entry point: runs the full 4-step address quality pipeline
from a single geocoded patients CSV file.

Usage:
    python run_pipeline.py <path_to_geocoded_patients_csv> [OPTIONS]

Positional argument:
    input_file        Path to the raw geocoded patients CSV (semicolon-separated).
                      Expected columns: RUE, CPVILLE, VILLE,
                      result_housenumber, result_name, result_postcode, result_city,
                      latitude, longitude.

Options:
    --gpu             CUDA device ID for GPU-accelerated steps (default: 2)
    --gpu-03          CUDA device ID for step 03 (default: same as --gpu)
    --output-dir      Root output directory (default: ./data/outputs/)
    --odonymes        Path to odonymes.txt (default: ./data/odonymes.txt)
    --prenoms         Path to Prenoms.csv (default: ./data/Prenoms.csv)
    --chunk-size      NW alignment chunk size for step 01 (default: 1000)
    --hospital-lat    Hospital latitude for hotspot step (default: 48.83931, HEGP)
    --hospital-lon    Hospital longitude for hotspot step (default: 2.27537, HEGP)
    --reference-file  Reference CSV for step 03 (auto-detected if omitted)
    --sample-size     Number of reference rows to sample in step 03 (default: 15000)
    --iris-file       IRIS spatial file for step 04 (default: ./data/socioeco/iris.geojson)
    --revenus-file    IRIS income CSV for step 04 (default: ./data/socioeco/BASE_TD_FILO_DISP_IRIS_2020.csv)
    --steps           Comma-separated list of steps to run (default: 1,2,3,4)
                      Example: --steps 1,2  (only run steps 01 and 02)

Examples:
    # Full pipeline
    python run_pipeline.py ../01_data/patients/patients_geocoded/patients_geocoded_20250311.csv

    # Steps 1 and 2 only, on GPU 3
    python run_pipeline.py patients.csv --gpu 3 --steps 1,2

    # Resume from step 3 using existing outputs
    python run_pipeline.py patients.csv --steps 3,4
"""

import argparse
import subprocess
import sys
import os
import glob
from datetime import datetime


def latest_file(pattern: str) -> str | None:
    """Return the most recently modified file matching glob *pattern*, or None."""
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def run_step(script: str, cmd_args: list[str], step_num: int) -> None:
    print(f"\n{'='*70}")
    print(f"  STEP {step_num:02d}  —  {os.path.basename(script)}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, script] + cmd_args,
        check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Full address quality pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input_file",      help="Path to the geocoded patients CSV.")
    parser.add_argument("--gpu",           type=str,   default="2")
    parser.add_argument("--gpu-03",        type=str,   default=None,
                        help="GPU device for step 03 (defaults to --gpu)")
    parser.add_argument("--output-dir",    default="./data/outputs/")
    parser.add_argument("--odonymes",      default="./data/odonymes.txt")
    parser.add_argument("--prenoms",       default="./data/Prenoms.csv")
    parser.add_argument("--chunk-size",      type=int,   default=1000)
    parser.add_argument("--min-bias-count",  type=int,   default=2,
                        help="Min token frequency for bias detection (default: 2). "
                             "Use 1 on small test datasets such as sample.csv.")
    parser.add_argument("--hospital-lat",  type=float, default=48.83931098867617)
    parser.add_argument("--hospital-lon",  type=float, default=2.2753718727661623)
    parser.add_argument("--reference-file",default=None)
    parser.add_argument("--sample-size",   type=int,   default=15000)
    parser.add_argument("--iris-file",     default="./data/socioeco/iris.geojson")
    parser.add_argument("--revenus-file",
                    default="./data/socioeco/BASE_TD_FILO_DISP_IRIS_2020.csv")    
    parser.add_argument("--steps",         default="1,2,3,4",
                        help="Comma-separated steps to execute (default: 1,2,3,4)")
    args = parser.parse_args()

    steps_to_run = {int(s.strip()) for s in args.steps.split(",")}
    gpu_03 = args.gpu_03 or args.gpu

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = datetime.now()
    print(f"Pipeline started at {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file : {args.input_file}")
    print(f"Steps      : {sorted(steps_to_run)}")
    print(f"GPU        : {args.gpu}  (step 03: {gpu_03})")

    # ── Step 01 ───────────────────────────────────────────────────────────────
    if 1 in steps_to_run:
        run_step(
            os.path.join(scripts_dir, "01_textbiais_identification.py"),
            [
                args.input_file,
                "--gpu",            args.gpu,
                "--output-dir",     args.output_dir,
                "--odonymes",       args.odonymes,
                "--prenoms",        args.prenoms,
                "--chunk-size",     str(args.chunk_size),
                "--min-bias-count", str(args.min_bias_count),
            ],
            step_num=1
        )

    # ── Step 02 ───────────────────────────────────────────────────────────────
    if 2 in steps_to_run:
        # Auto-detect the wstreetInfos CSV from step 01 output
        street_csv = latest_file(os.path.join(args.output_dir, "*_wstreetInfos.csv"))
        if street_csv is None:
            raise FileNotFoundError(
                f"No '*_wstreetInfos.csv' found in {args.output_dir}. "
                "Run step 01 first, or pass the file explicitly."
            )
        print(f"\n[run_pipeline] Step 02 input: {street_csv}")
        run_step(
            os.path.join(scripts_dir, "02_hotspot_identification.py"),
            [
                street_csv,
                "--output-dir",    args.output_dir,
                "--hospital-lat",  str(args.hospital_lat),
                "--hospital-lon",  str(args.hospital_lon),
            ],
            step_num=2
        )

    # ── Step 03 ───────────────────────────────────────────────────────────────
    if 3 in steps_to_run:
        # Guard: skip step 03 entirely if no biases were identified in step 01
        biaises_file = "./data/biaises_identified.csv"
        biaises_empty = True
        if os.path.exists(biaises_file):
            try:
                import pandas as _pd
                _df = _pd.read_csv(biaises_file, delimiter="|")
                biaises_empty = len(_df.dropna()) == 0
            except Exception:
                biaises_empty = True

        if biaises_empty:
            print(
                "\n[run_pipeline] Step 03 SKIPPED — biaises_identified.csv is empty.\n"
                "  No biases to inject into the reference dataset.\n"
                "  Re-run step 01 with a lower --min-bias-count value, or check that\n"
                "  the input data contains addresses with extra unmatched tokens."
            )
        else:
            print(f"\n[run_pipeline] Step 03 — injecting biases from biaises_identified.csv")
            step03_args = []
            if args.reference_file:
                step03_args += ["--reference-file", args.reference_file]
            step03_args += ["--sample-size", str(args.sample_size)]
            run_step(
                os.path.join(scripts_dir, "03_ref_biaised_integration.py"),
                step03_args,
                step_num=3
            )

    # ── Step 04 ───────────────────────────────────────────────────────────────
    if 4 in steps_to_run:
        # Guard: skip step 04 if step 03 produced no geocoded bias files
        geocoded_dir = "./data/reference/ref_biaised_geocoded/"
        has_geocoded = (
            os.path.isdir(geocoded_dir) and
            any(f.endswith(".csv") for f in os.listdir(geocoded_dir))
        ) if not biaises_empty else False

        if not has_geocoded:
            print(
                "\n[run_pipeline] Step 04 spatial analysis SKIPPED — no geocoded bias "
                "files found.\n"
                "  Steps 01–02 outputs (alignment, hotspot detection, quality breakdown)\n"
                "  are still available in data/outputs/."
            )
        else:
            classified_csv = latest_file(
                os.path.join(args.output_dir, "*_clean_classified.csv")
            )
            if classified_csv is None:
                raise FileNotFoundError(
                    f"No '*_clean_classified.csv' found in {args.output_dir}."
                )
            hotspot_csv = latest_file(os.path.join(args.output_dir, "*hotspot*.csv"))
            print(f"\n[run_pipeline] Step 04 input: {classified_csv}")
            step04_args = [
                classified_csv,
                "--output-dir",   args.output_dir,
                "--iris-file",    args.iris_file,
                "--revenus-file", args.revenus_file,
            ]
            if hotspot_csv:
                step04_args += ["--hotspot-file", hotspot_csv]
            run_step(
                os.path.join(scripts_dir, "04_results_analysis.py"),
                step04_args,
                step_num=4
            )

    elapsed = datetime.now() - t_start
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {elapsed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()