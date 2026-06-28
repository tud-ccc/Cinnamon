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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm

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

# ── Config filter ─────────────────────────────────────────────────────────────
# Edit this function to restrict which configs are compiled and benchmarked.
# Receives the config parameters as a dict {param_name: int_value}.
# Return True to include the config, False to skip it.

def config_filter(params: dict) -> bool:
    return params['mramCol'] * params['dpus'] >= 128000 and 4 <= params['dpus'] <= 512

# ── Pool parsing ─────────────────────────────────────────────────────────────

def parse_pool(pool_csv: pathlib.Path):
    """Return list of (row_index, params_dict, num_dpus) for rows where valid == 1 and config_filter passes."""
    configs = []
    with open(pool_csv) as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in NON_PARAM_COLS]
        for i, row in enumerate(reader):
            if row.get("valid", "0").strip() != "1":
                continue
            try:
                params = {c: int(row[c]) for c in param_cols}
            except (ValueError, KeyError) as e:
                print(f"  skip row {i}: {e}", file=sys.stderr)
                continue
            if not config_filter(params):
                continue
            configs.append((i, params, params.get("dpus", 1)))
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
def fmt_cmd(args: list[str]) -> str:
    def quote(s):
      return f'"{s}"' if ' ' in s else s
    return ' '.join(quote(s) for s in args)

def compile_one(args):
    (fn_name, config_id, params, fn_module_path,
     run_dir, makefile_dir, cinm_opt, pre_passes) = args

    config_dir = pathlib.Path(run_dir) / fn_name / f"config_{config_id:05d}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write a one-row CSV with the config parameters so the aggregation script
    # can join it with benchmark output without re-reading the pool.
    config_csv = config_dir / "config.csv"
    if not config_csv.exists():
        with open(config_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fn_name", "config_id"] + list(params.keys()))
            writer.writeheader()
            writer.writerow({"fn_name": fn_name, "config_id": config_id, **params})

    # Step 1: lower the single-function module with this configuration.
    lowered = config_dir / "lowered.mlir"
    solution_str = ",".join(str(v) for v in params.values())
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
        cinm_opt_cmd = fmt_cmd(cmd_opt)
        (config_dir / "cinm_opt_stderr.txt").write_text(r.stderr)
        (config_dir / "failed").touch()
        return fn_name, config_id, False, f"cinm-opt failed:\n{cinm_opt_cmd}\n{r.stderr}"

    # Step 2: compile host + DPU.
    ir_dir  = config_dir / "ir"
    bin_dir = config_dir / "bin"
    cmd_make = [
        "make", "-C", str(makefile_dir),
        f"SRC_MLIR={lowered.resolve()}",
        f"IR_DIR={ir_dir.resolve()}",
        f"BIN_DIR={bin_dir.resolve()}",
        f"BENCH_FN={fn_name}",
        f"BENCH_N={FUNC_SIZES[fn_name]}",
        "bench-single"
    ]
    r = subprocess.run(cmd_make, capture_output=True, text=True)
    if r.returncode != 0:
        (config_dir / "make_stderr.txt").write_text(r.stderr)
        (config_dir / "failed").touch()
        return fn_name, config_id, False, f"make failed:\n{r.stderr}"

    return fn_name, config_id, True, ""


# ── Run step (sequential to avoid DPU over-allocation) ───────────────────────

def run_one(fn_name, config_id, run_dir, iters):
    config_dir = pathlib.Path(run_dir).absolute() / fn_name / f"config_{config_id:05d}"
    bench_bin  = config_dir / "bin" / f"bench_{fn_name}"
    output_dir = config_dir / "output"
    output_dir.mkdir(exist_ok=True)

    cmd = [str(bench_bin), str(output_dir), str(iters)]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # DPU binaries are loaded relative to cwd; place bench next to them.
        cwd=str(config_dir / "bin" / fn_name),
    )
    if r.returncode != 0:
        (config_dir / "bench_stderr.txt").write_text(r.stderr)
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
    parser.add_argument("--fn",           default=None,
                        help="Only process this function (e.g. red_4MB)")
    parser.add_argument("--dpu-cap",    type=int, default=1024,
                        help="Max DPUs to use concurrently during benchmarking (default 2048)")
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
        if args.fn is not None and fn_name != args.fn:
            continue
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
        for config_id, vals, num_dpus in configs:
            tasks.append((fn_name, config_id, vals, num_dpus))

    print(f"\nTotal: {len(tasks)} (function, config) pairs")

    # ── Compile phase ─────────────────────────────────────────────────────────
    compiled = []
    if not args.run_only:
        compile_args = [
            (fn, cid, params, str(modules[fn]),
             str(run_dir), str(here), args.cinm_opt, PRE_PASSES)
            for fn, cid, params, _num_dpus in tasks
        ]
        print(f"\nCompiling with {args.workers} workers...")
        dpus_for = {(fn, cid): nd for fn, cid, _, nd in tasks}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(compile_one, a): (a[0], a[1]) for a in compile_args}
            for fut in tqdm(as_completed(futures), desc="Compiling", total=len(futures)):
                fn_name, cid = futures[fut]
                _, _, ok, msg = fut.result()
                status = "OK" if ok else "FAIL"
                tqdm.write(f"  [{status}] {fn_name} config {cid:05d}")
                if not ok:
                    tqdm.write(f"         {msg}", file=sys.stderr)
                if ok:
                    compiled.append((fn_name, cid, dpus_for[fn_name, cid]))
    else:
        compiled = [
            (fn, cid, nd) for fn, cid, _, nd in tasks
            if not (run_dir / fn / f"config_{cid:05d}" / "failed").exists()
        ]

    # ── Benchmark phase ───────────────────────────────────────────────────────
    if not args.compile_only:
        print(f"\nRunning {len(compiled)} benchmarks "
              f"(parallel, DPU cap={args.dpu_cap})...")
        pending   = list(compiled)   # [(fn_name, cid, num_dpus), ...]
        in_flight = {}               # future -> (fn_name, cid, num_dpus)
        used_dpus = 0

        with ThreadPoolExecutor(max_workers=len(pending) or 1) as ex:
            pbar = tqdm(total=len(pending), desc="Running",)
            while pending or in_flight:
                # Launch every task that fits within the DPU cap.
                remaining = []
                for fn_name, cid, num_dpus in pending:
                    if used_dpus + num_dpus <= args.dpu_cap:
                        fut = ex.submit(run_one, fn_name, cid, str(run_dir), args.iters)
                        in_flight[fut] = (fn_name, cid, num_dpus)
                        used_dpus += num_dpus
                    else:
                        remaining.append((fn_name, cid, num_dpus))
                pending = remaining

                if not in_flight:
                    # A task needs more DPUs than the cap — launch it alone.
                    fn_name, cid, num_dpus = pending.pop(0)
                    tqdm.write(f"  WARNING: {fn_name} config {cid:05d} needs "
                               f"{num_dpus} DPUs > cap {args.dpu_cap}, running alone",
                               file=sys.stderr)
                    fut = ex.submit(run_one, fn_name, cid, str(run_dir), args.iters)
                    in_flight[fut] = (fn_name, cid, num_dpus)
                    used_dpus += num_dpus

                # Wait for at least one to finish before re-scheduling.
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    fn_name, cid, num_dpus = in_flight.pop(fut)
                    used_dpus -= num_dpus
                    _, _, ok, msg = fut.result()
                    status = "OK" if ok else "FAIL"
                    tqdm.write(f"  [{status}] {fn_name} config {cid:05d}"
                               f"  (used={used_dpus}/{args.dpu_cap} DPUs)")
                    if not ok:
                        tqdm.write(f"         {msg[:300]}", file=sys.stderr)
                    pbar.update(1)
            pbar.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
