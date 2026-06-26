#!/usr/bin/env python3
"""
Aggregate benchmark output CSVs from a run directory into per-type dataframes.

Usage:
  python3 aggregate.py --run-dir runs/ --out aggregated/

Walks every {run_dir}/{fn_name}/config_*/  directory that contains both
config.csv and an output/ subdirectory, and produces one output CSV per
measurement type (scatter, gather, launch, free, total).

Each output CSV contains the original measurement columns plus fn_name,
config_id, and all parameter columns from config.csv, so the full dataset
can be grouped or filtered by any dimension without joins.
"""

import argparse
import pathlib
import sys

import pandas as pd


def iter_config_dirs(run_dir: pathlib.Path):
    """Yield (fn_name, config_dir) for every valid config directory."""
    for fn_dir in sorted(run_dir.iterdir()):
        if not fn_dir.is_dir() or fn_dir.name.startswith("_"):
            continue
        for config_dir in sorted(fn_dir.iterdir()):
            if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
                continue
            yield fn_dir.name, config_dir


def csv_type(path: pathlib.Path) -> str:
    """Extract measurement type from filename, e.g. red_256MB_gather.csv → gather."""
    return path.stem.rsplit("_", 1)[-1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="Root run directory (contains fn_name/config_*/ subdirs)")
    parser.add_argument("--out", required=True,
                        help="Output directory for aggregated CSVs")
    parser.add_argument("--types", default=None,
                        help="Comma-separated list of types to include "
                             "(default: all found, e.g. scatter,gather,launch,free,total)")
    args = parser.parse_args()

    run_dir  = pathlib.Path(args.run_dir)
    out_dir  = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    type_filter = set(args.types.split(",")) if args.types else None

    frames: dict[str, list[pd.DataFrame]] = {}
    n_configs = 0
    n_missing = 0

    for fn_name, config_dir in iter_config_dirs(run_dir):
        config_csv  = config_dir / "config.csv"
        output_dir  = config_dir / "output"

        if not config_csv.exists():
            print(f"  skip {config_dir.relative_to(run_dir)}: no config.csv",
                  file=sys.stderr)
            n_missing += 1
            continue
        if not output_dir.exists() or not any(output_dir.iterdir()):
            continue

        config_meta = pd.read_csv(config_csv).iloc[0].to_dict()
        n_configs += 1

        for csv_path in sorted(output_dir.glob("*.csv")):
            t = csv_type(csv_path)
            if type_filter and t not in type_filter:
                continue

            df = pd.read_csv(csv_path)
            for col, val in config_meta.items():
                df[col] = val

            frames.setdefault(t, []).append(df)

    if n_missing:
        print(f"  {n_missing} config dir(s) skipped (no config.csv — "
              f"re-run compile phase to generate them)", file=sys.stderr)

    if not frames:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    for t, dfs in sorted(frames.items()):
        out_path = out_dir / f"{t}.csv"
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"  {t:10s}  {len(combined):>8,} rows  →  {out_path}")

    print(f"\n{n_configs} configs aggregated into {len(frames)} file(s).")


if __name__ == "__main__":
    main()
