#!/usr/bin/env python3
"""
Aggregate per-seed BO-search timings from @results into a single CSV.

Walks {results_dir}/infer_{fn_name}/seed_{N}/timings.csv, adds fn_name and
seed columns, and writes a merged timings.csv to {out_dir}/timings.csv.

Usage:
  python3 aggregate_bo_timings.py --results-dir @results --out @+bo_timings
"""

import argparse
import pathlib
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Path to the @results directory")
    parser.add_argument("--out", required=True,
                        help="Output directory; timings.csv is written here")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fn_dir in sorted(results_dir.iterdir()):
        if not fn_dir.is_dir() or not fn_dir.name.startswith("infer_"):
            continue
        fn_name = fn_dir.name[len("infer_"):]
        for seed_dir in sorted(fn_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name[len("seed_"):])
            csv = seed_dir / "timings.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df.insert(0, "fn_name", fn_name)
            df.insert(1, "seed", seed)
            frames.append(df)

    if not frames:
        print(f"ERROR: no timings.csv found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / "timings.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Wrote {len(merged)} rows ({merged['fn_name'].nunique()} functions, "
          f"{merged['seed'].nunique()} seeds) → {out_csv}")


if __name__ == "__main__":
    main()
