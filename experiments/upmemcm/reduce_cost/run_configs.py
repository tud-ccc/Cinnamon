#!/usr/bin/env python3
"""
Build and benchmark every valid configuration found in a dump-dir.

Usage:
  python3 run_configs.py \\
      --data  experiments/data/prim_red_oracle \\
      --src   experiments/prim_red.mlir \\
      [--workers 4] [--iters 5] [--run-dir runs/] [--compile-only] [--run-only]

Layout of --data:
  {data}/infer_{fn_name}/pool.csv    (one per function)

For each (function, valid config) pair the script:
  1. Runs cinm-opt on the single-function module with eval-solution=...
  2. Runs `make bench-single BENCH_FN=<fn> ...` to compile
  3. Runs the resulting binary to collect scatter/gather/launch CSVs
"""

import argparse
import csv
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Constants ────────────────────────────────────────────────────────────────

# Element count for each function (i32 elements).
FUNC_SIZES = {
    "red_4MB":   524288,
    "red_64MB":  8388608,
    "red_256MB": 34554432,
    "red_512MB": 67108864,
}

# Columns in pool.csv that are NOT search parameters.
NON_PARAM_COLS = frozenset({
    "visited", "valid", "cost", "eval_iter", "eval_time_ms",
    "mu", "sigma", "acq", "index",
})

# Passes to run before upmem-infer-accelerator.
PRE_PASSES = ["--cinm-assign-platforms", "--cinm-isolate-compute-blocks"]

# ── Pool parsing ─────────────────────────────────────────────────────────────

def parse_pool(pool_csv: pathlib.Path):
    """Return list of (row_index, list[int]) for rows where valid == 1."""
    configs = []
    with open(pool_csv) as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in NON_PARAM_COLS]
        for i, row in enumerate(reader):
            if row.get("valid", "0").strip() != "1":
                continue
            try:
                values = [int(row[c]) for c in param_cols]
            except (ValueError, KeyError) as e:
                print(f"  skip row {i}: {e}", file=sys.stderr)
                continue
            configs.append((i, values))
    return configs


def find_function_pools(data_dir: pathlib.Path):
    """Yield (fn_name, pool_csv_path) for every infer_* subdir with a pool.csv."""
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        pool = sub / "pool.csv"
        if not pool.exists():
            continue
        # Strip leading "infer_" to get the MLIR function name.
        fn_name = sub.name.removeprefix("infer_")
        yield fn_name, pool


# ── Source MLIR splitting ────────────────────────────────────────────────────

def split_source(src_mlir: pathlib.Path, out_dir: pathlib.Path):
    """Split src_mlir on // ----- and write one file per non-empty chunk.

    Returns a dict {fn_name: Path} for each function found.
    """
    text = src_mlir.read_text()
    chunks = [c.strip() for c in text.split("\n// -----\n")]
    modules = {}
    for chunk in chunks:
        if not chunk or chunk.startswith("module") and len(chunk) < 20:
            continue  # skip empty module preamble
        # Derive function name from `func.func @name`.
        for line in chunk.splitlines():
            line = line.strip()
            if line.startswith("func.func @"):
                fn = line.split("@")[1].split("(")[0]
                path = out_dir / f"{fn}.mlir"
                path.write_text(chunk)
                modules[fn] = path
                break
    return modules


# ── Compile step (runs in a worker process) ──────────────────────────────────

def compile_one(args):
    (fn_name, config_id, param_values, fn_module_path,
     run_dir, makefile_dir, cinm_opt, pre_passes) = args

    config_dir = pathlib.Path(run_dir) / fn_name / f"config_{config_id:05d}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: lower the single-function module with this configuration.
    lowered = config_dir / "lowered.mlir"
    solution_str = ",".join(str(v) for v in param_values)
    cmd_opt = [
        cinm_opt,
        str(fn_module_path),
        "--split-input-file",
        *pre_passes,
        f"--upmem-infer-accelerator=eval-solution={solution_str}",
        "-o", str(lowered),
    ]
    r = subprocess.run(cmd_opt, capture_output=True, text=True)
    if r.returncode != 0:
        return fn_name, config_id, False, f"cinm-opt failed:\n{r.stderr[-3000:]}"

    # Step 2: compile host + DPU.
    ir_dir  = config_dir / "ir"
    bin_dir = config_dir / "bin"
    cmd_make = [
        "make", "-C", str(makefile_dir),
        f"SRC_MLIR={lowered.resolve()}",
        f"IR_DIR={ir_dir.resolve()}",
        f"BIN_DIR={bin_dir.resolve()}",
        f"BENCH_FN={fn_name}",
        "bench-single",
    ]
    r = subprocess.run(cmd_make, capture_output=True, text=True)
    if r.returncode != 0:
        return fn_name, config_id, False, f"make failed:\n{r.stderr[-3000:]}"

    return fn_name, config_id, True, ""


# ── Run step (sequential to avoid DPU over-allocation) ───────────────────────

def run_one(fn_name, config_id, run_dir, fn_size, iters):
    config_dir = pathlib.Path(run_dir) / fn_name / f"config_{config_id:05d}"
    bench_bin  = config_dir / "bin" / f"bench_{fn_name}"
    output_dir = config_dir / "output"
    output_dir.mkdir(exist_ok=True)

    cmd = [str(bench_bin), str(fn_size), str(output_dir), str(iters)]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # DPU binaries are loaded relative to cwd; place bench next to them.
        cwd=str(config_dir / "bin"),
    )
    if r.returncode != 0:
        return fn_name, config_id, False, r.stderr[-1000:]
    return fn_name, config_id, True, r.stdout.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    here = pathlib.Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",    required=True,
                        help="Directory containing infer_*/pool.csv files")
    parser.add_argument("--src",     required=True,
                        help="High-level cinm source MLIR (with // ----- splits)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel compile workers (default 4)")
    parser.add_argument("--iters",   type=int, default=5,
                        help="Benchmark iterations per config (default 5)")
    parser.add_argument("--run-dir", default="runs",
                        help="Root output directory (default: runs/)")
    parser.add_argument("--cinm-opt",
                        default=str(here / "../../../build/bin/cinm-opt"))
    parser.add_argument("--limit",        type=int, default=None,
                        help="Only process the first N configs per function (for testing)")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only",     action="store_true")
    args = parser.parse_args()

    data_dir   = pathlib.Path(args.data)
    src_mlir   = pathlib.Path(args.src)
    run_dir    = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-split the source MLIR once.
    split_dir = run_dir / "_split"
    split_dir.mkdir(exist_ok=True)
    modules = split_source(src_mlir, split_dir)
    print(f"Split {src_mlir.name} → {list(modules.keys())}")

    # Collect (fn_name, config_id, param_values) tasks.
    tasks = []
    for fn_name, pool_csv in find_function_pools(data_dir):
        if fn_name not in modules:
            print(f"  WARNING: {fn_name} not found in source MLIR, skipping", file=sys.stderr)
            continue
        if fn_name not in FUNC_SIZES:
            print(f"  WARNING: unknown size for {fn_name}, skipping", file=sys.stderr)
            continue
        configs = parse_pool(pool_csv)
        if args.limit is not None:
            configs = configs[:args.limit]
        print(f"  {fn_name}: {len(configs)} valid configs in {pool_csv.parent.name}")
        for config_id, vals in configs:
            tasks.append((fn_name, config_id, vals))

    print(f"\nTotal: {len(tasks)} (function, config) pairs")

    # ── Compile phase ─────────────────────────────────────────────────────────
    compiled = []
    if not args.run_only:
        compile_args = [
            (fn, cid, vals, str(modules[fn]),
             str(run_dir), str(here), args.cinm_opt, PRE_PASSES)
            for fn, cid, vals in tasks
        ]
        print(f"\nCompiling with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(compile_one, a): (a[0], a[1]) for a in compile_args}
            for fut in as_completed(futures):
                fn_name, cid = futures[fut]
                _, _, ok, msg = fut.result()
                status = "OK" if ok else "FAIL"
                print(f"  [{status}] {fn_name} config {cid:05d}")
                if not ok:
                    print(f"         {msg[:300]}", file=sys.stderr)
                if ok:
                    compiled.append((fn_name, cid))
    else:
        compiled = [(fn, cid) for fn, cid, _ in tasks]

    # ── Benchmark phase ───────────────────────────────────────────────────────
    if not args.compile_only:
        print(f"\nRunning {len(compiled)} benchmarks (sequential)...")
        for fn_name, cid in compiled:
            fn_size = FUNC_SIZES[fn_name]
            _, _, ok, msg = run_one(fn_name, cid, run_dir, fn_size, args.iters)
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {fn_name} config {cid:05d}")
            if not ok:
                print(f"         {msg[:300]}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
