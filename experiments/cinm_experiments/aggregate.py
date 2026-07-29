"""Merge per-config output into per-source summary CSVs, for experiments
that sweep many (function, seed) configs and need one combined view per
measurement type instead of thousands of tiny per-config files -- e.g.
experiments/paperplots, which runs O(1000) (function, seed) configs per
source."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd


def _csv_type(path: pathlib.Path) -> str:
    """Extract measurement type from filename, e.g. red_256MB_gather.csv -> gather."""
    return path.stem.rsplit("_", 1)[-1]


def iter_config_dirs(run_dir: pathlib.Path):
    """Yield (fn_name, config_dir) for every config directory under run_dir
    (run_dir/{fn_name}/{config_id}/), skipping non-directories and any
    fn_name starting with "_" (e.g. a stray _split/)."""
    run_dir = pathlib.Path(run_dir)
    for fn_dir in sorted(run_dir.iterdir()):
        if not fn_dir.is_dir() or fn_dir.name.startswith("_"):
            continue
        for config_dir in sorted(fn_dir.iterdir()):
            if config_dir.is_dir():
                yield fn_dir.name, config_dir


def aggregate_run(
    run_dir: pathlib.Path,
    compile_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    types: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Merge every config's output/*.csv (scatter/gather/alloc/free/total/...,
    written by a bench_* binary) across run_dir into one combined DataFrame
    per measurement type, tagged with that config's fn_name and every column
    from its config.csv (so the result can be grouped/filtered by any config
    parameter with no joins). Writes {out_dir}/{type}.csv per type found and
    returns the same frames as a dict. Skips configs with no config.csv (not
    compiled) or an empty/missing output/ (not run)."""
    run_dir = pathlib.Path(run_dir)
    compile_dir = pathlib.Path(compile_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, list[pd.DataFrame]] = {}
    n_configs = n_missing = 0

    for fn_name, config_dir in iter_config_dirs(run_dir):
        config_csv = compile_dir / fn_name / config_dir.name / "config.csv"
        output_dir = config_dir / "output"

        if not config_csv.exists():
            n_missing += 1
            continue
        if not output_dir.exists() or not any(output_dir.iterdir()):
            continue

        config_meta = pd.read_csv(config_csv).iloc[0].to_dict()
        n_configs += 1

        for csv_path in sorted(output_dir.glob("*.csv")):
            t = _csv_type(csv_path)
            if types and t not in types:
                continue
            df = pd.read_csv(csv_path)
            for col, val in config_meta.items():
                df[col] = val
            frames.setdefault(t, []).append(df)

    if n_missing:
        print(
            f"  {n_missing} config dir(s) skipped (no config.csv -- not compiled)",
            file=sys.stderr,
        )
    if not frames:
        raise RuntimeError(f"no aggregatable data found under {run_dir}")

    combined = {}
    for t, dfs in sorted(frames.items()):
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out_dir / f"{t}.csv", index=False)
        combined[t] = df
        print(f"  {t:10s}  {len(df):>8,} rows  -> {out_dir / f'{t}.csv'}")

    print(f"{n_configs} configs aggregated into {len(frames)} file(s).")
    return combined


def aggregate_predicted_costs(
    compile_dir: pathlib.Path, out_dir: pathlib.Path
) -> pd.DataFrame:
    """Merge every config's ir/cost.csv (block_id/location/category/label/
    cost_ms/block_total_ms -- the cost-model breakdown written during
    compile by UpmemAnnotateCosts, see cinmopt.eval_solution_lowerer) across
    compile_dir into one predicted_costs.csv, tagged with fn_name and every
    column from that config's config.csv -- same join key (fn_name/label/
    params) aggregate_run uses on the measured side, so the two can be
    joined later without any extra bookkeeping.

    compile_dir is expected in aggregate_run's compile_dir shape
    ({fn_name}/{label}/, e.g. PATHS.compile_root(prim) / SYSTEM), since
    everything read here (config.csv, ir/cost.csv) lives under compile_dir
    already -- no separate run_dir needed.

    cost.csv's own `label` column (the block label, e.g. "kernel",
    "launchOverhead") is renamed to `cost_label` before merging in the
    config's metadata, since config.csv also has a `label` column (the
    config's row id, e.g. "row_00157") -- without the rename, assigning the
    config's `label` column would silently clobber the block label.

    Skips configs with no config.csv (not compiled) or no ir/cost.csv (e.g.
    a plugin/simulator that doesn't emit a per-op breakdown)."""
    compile_dir = pathlib.Path(compile_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    n_configs = n_missing = 0

    for fn_name, config_dir in iter_config_dirs(compile_dir):
        config_csv = config_dir / "config.csv"
        cost_csv = config_dir / "ir" / "cost.csv"

        if not config_csv.exists() or not cost_csv.exists():
            n_missing += 1
            continue

        config_meta = pd.read_csv(config_csv).iloc[0].to_dict()
        n_configs += 1

        df = pd.read_csv(cost_csv).rename(columns={"label": "cost_label"})
        for col, val in config_meta.items():
            df[col] = val
        frames.append(df)

    if n_missing:
        print(
            f"  {n_missing} config dir(s) skipped (no config.csv or ir/cost.csv)",
            file=sys.stderr,
        )
    if not frames:
        raise RuntimeError(f"no predicted cost data found under {compile_dir}")

    combined = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / "predicted_costs.csv"
    combined.to_csv(out_csv, index=False)
    print(f"  {n_configs} configs aggregated -> {out_csv} ({len(combined):,} rows)")
    return combined


def aggregate_bo_timings(
    results_dir: pathlib.Path, out_dir: pathlib.Path
) -> pd.DataFrame:
    """Merge every {results_dir}/infer_{fn_name}/seed_{N}/timings.csv (raw
    per-seed BO-search timing, written by cinmopt.bo_multiseed) into one
    {out_dir}/timings.csv tagged with fn_name and seed."""
    results_dir = pathlib.Path(results_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fn_dir in sorted(results_dir.iterdir()):
        if not fn_dir.is_dir() or not fn_dir.name.startswith("infer_"):
            continue
        fn_name = fn_dir.name.removeprefix("infer_")
        for seed_dir in sorted(fn_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.removeprefix("seed_"))
            csv_path = seed_dir / "timings.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df.insert(0, "fn_name", fn_name)
            df.insert(1, "seed", seed)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"no timings.csv found under {results_dir}")

    merged = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / "timings.csv"
    merged.to_csv(out_csv, index=False)
    print(
        f"  {len(merged)} rows ({merged['fn_name'].nunique()} functions, "
        f"{merged['seed'].nunique()} seeds) -> {out_csv}"
    )
    return merged
