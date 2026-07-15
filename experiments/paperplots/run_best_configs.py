#!/usr/bin/env python3
"""
For each seed in a results directory, find the best visited config and benchmark it.

Usage:
  python3 run_best_configs.py \\
      --results      .experiments/tags/prim_red_cinm2_hybrid400/results \\
      --src          experiments/prim_red.mlir \\
      --compile-dir  compiled/ \\
      --run-dir      runs/ \\
      [--workers 4] [--iters 5] [--compile-only] [--run-only]

Layout of --results:
  {results}/{problem}/seed_{N}/pool.csv

For each (problem, seed) the script picks the row with visited==1 and minimal cost,
then follows the same compile+run pipeline as run_configs.py.

--compile-dir holds lowered MLIR, IR, and binaries (written by compile phase).
--run-dir     holds benchmark output CSVs (written by run phase).
Both have the same {problem}/seed_{N}/ structure.
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

NON_PARAM_COLS = frozenset({
    "visited", "valid", "cost", "eval_iter", "eval_time_ms", "cpu_time_ms",
    "mu", "sigma", "acq", "index",
})

PRE_PASSES = ["--cinm-assign-platforms", "--cinm-isolate-compute-blocks"]

# ── Pool parsing ─────────────────────────────────────────────────────────────

def parse_best(pool_csv: pathlib.Path):
    """Return (params_dict, num_dpus) for the visited row with the lowest cost,
    or None if no visited row with a parseable cost exists."""
    best_cost = None
    best_params = None
    with open(pool_csv) as f:
        reader = csv.DictReader(f)
        param_cols = [c for c in reader.fieldnames if c not in NON_PARAM_COLS]
        for i, row in enumerate(reader):
            if row.get("visited", "0").strip() != "1":
                continue
            cost_str = row.get("cost", "").strip()
            if not cost_str:
                continue
            try:
                cost = float(cost_str)
                params = {c: int(row[c]) for c in param_cols}
            except (ValueError, KeyError) as e:
                print(f"  skip row {i} in {pool_csv}: {e}", file=sys.stderr)
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_params = params
    if best_params is None:
        return None
    return best_params, best_params.get("dpus", 1)


def find_seed_pools(results_dir: pathlib.Path):
    """Yield (fn_name, seed, pool_csv_path) for every problem/seed subdir."""
    for problem_dir in sorted(results_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        fn_name = problem_dir.name.removeprefix("infer_")
        for seed_dir in sorted(problem_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            pool = seed_dir / "pool.csv"
            if not pool.exists():
                continue
            seed = seed_dir.name.removeprefix("seed_")
            yield fn_name, seed, pool


# ── Source MLIR splitting ────────────────────────────────────────────────────

def split_source(src_mlir: pathlib.Path, out_dir: pathlib.Path):
    text = src_mlir.read_text()
    chunks = [c.strip() for c in text.split("\n// -----\n")]
    modules = {}
    for chunk in chunks:
        if not chunk or chunk.startswith("module") and len(chunk) < 20:
            continue
        for line in chunk.splitlines():
            line = line.strip()
            if line.startswith("func.func @"):
                fn = line.split("@")[1].split("(")[0]
                path = out_dir / f"{fn}.mlir"
                path.write_text(chunk)
                modules[fn] = path
                break
    return modules


# ── Compile step ─────────────────────────────────────────────────────────────

def fmt_cmd(args: list[str]) -> str:
    def quote(s):
        return f'"{s}"' if ' ' in s else s
    return ' '.join(quote(s) for s in args)


def compile_one(args):
    (fn_name, seed, params, fn_module_path,
     compile_dir, makefile_dir, cinm_opt, pre_passes, prim) = args

    config_dir = pathlib.Path(compile_dir) / fn_name / f"seed_{seed}"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_csv = config_dir / "config.csv"
    if not config_csv.exists():
        with open(config_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fn_name", "seed"] + list(params.keys()))
            writer.writeheader()
            writer.writerow({"fn_name": fn_name, "seed": seed, **params})

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
        (config_dir / "cinm_opt_stderr.txt").write_text(r.stderr)
        (config_dir / "failed").touch()
        return fn_name, seed, False, f"cinm-opt failed:\n{fmt_cmd(cmd_opt)}\n{r.stderr}"

    ir_dir  = config_dir / "ir"
    bin_dir = config_dir / "bin"
    cmd_make = [
        "make", "-C", str(makefile_dir),
        f"SRC_MLIR={lowered.resolve()}",
        f"IR_DIR={ir_dir.resolve()}",
        f"BIN_DIR={bin_dir.resolve()}",
        f"BENCH_FN={fn_name}",
        f"PRIM={prim}",
        "bench-single"
    ]
    r = subprocess.run(cmd_make, capture_output=True, text=True)
    if r.returncode != 0:
        (config_dir / "make_stderr.txt").write_text(r.stderr)
        (config_dir / "failed").touch()
        return fn_name, seed, False, f"make failed:\n{r.stderr}"

    return fn_name, seed, True, ""


# ── Run step ─────────────────────────────────────────────────────────────────

def run_one(fn_name, seed, compile_dir, run_dir, iters):
    compile_config_dir = pathlib.Path(compile_dir).absolute() / fn_name / f"seed_{seed}"
    run_config_dir     = pathlib.Path(run_dir).absolute() / fn_name / f"seed_{seed}"
    bench_bin  = compile_config_dir / "bin" / f"bench_{fn_name}"
    output_dir = run_config_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(bench_bin), str(output_dir), str(iters)]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(compile_config_dir / "bin" / fn_name),
    )
    if r.returncode != 0:
        (run_config_dir / "bench_stderr.txt").write_text(r.stderr)
        return fn_name, seed, False, r.stderr[-1000:]
    return fn_name, seed, True, r.stdout.strip()


def is_dpu_allocation_error(stderr: str) -> bool:
    return "allocation error" in stderr.lower()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    here = pathlib.Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir",  required=True,
                        help="Results directory containing {problem}/seed_{N}/pool.csv")
    parser.add_argument("--src",      required=True,
                        help="High-level cinm source MLIR (with // ----- splits)")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--iters",        type=int, default=5)
    parser.add_argument("--compile-dir",  required=True,
                        help="Directory for compiled artifacts (lowered MLIR, IR, binaries)")
    parser.add_argument("--run-dir",      default=None,
                        help="Directory for benchmark output CSVs (required unless --compile-only)")
    parser.add_argument("--cinm-opt", default=str(here / "../../build/bin/cinm-opt"))
    parser.add_argument("--problem",  default=None,
                        help="Only process this function (e.g. red_4MB)")
    parser.add_argument("--prim", default=None,
                        help="Primitive name (red, gemv, …); inferred from --src stem if omitted")
    parser.add_argument("--dpu-cap",  type=int, default=1024)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run-only",     action="store_true")
    args = parser.parse_args()

    if args.workers == 0:
      args.workers = os.cpu_count()

    if not args.compile_only and not args.run_dir:
        parser.error("--run-dir is required unless --compile-only is set")

    results_dir  = pathlib.Path(args.in_dir)
    src_mlir     = pathlib.Path(args.src)
    compile_dir  = pathlib.Path(args.compile_dir)
    run_dir      = pathlib.Path(args.run_dir) if args.run_dir else None
    compile_dir.mkdir(parents=True, exist_ok=True)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Infer prim from the MLIR filename (prim_red.mlir → "red") if not given.
    prim = args.prim or src_mlir.stem.removeprefix("prim_")

    split_dir = compile_dir / "_split"
    split_dir.mkdir(exist_ok=True)
    modules = split_source(src_mlir, split_dir)
    print(f"Split {src_mlir.name} → {list(modules.keys())}")

    tasks = []
    for fn_name, seed, pool_csv in find_seed_pools(results_dir):
        if args.problem is not None and fn_name != args.problem:
            continue
        if fn_name not in modules:
            print(f"  WARNING: {fn_name} not found in source MLIR, skipping", file=sys.stderr)
            continue
        result = parse_best(pool_csv)
        if result is None:
            print(f"  WARNING: no visited config in {pool_csv}, skipping", file=sys.stderr)
            continue
        params, num_dpus = result
        tasks.append((fn_name, seed, params, num_dpus))

    print(f"\nTotal: {len(tasks)} (function, seed) pairs")

    # ── Compile phase ─────────────────────────────────────────────────────────
    compiled = []
    if not args.run_only:
        makefile_dir = str(here / ".." / "upmemcm" / "reduce_cost")
        compile_args = [
            (fn, seed, params, str(modules[fn]),
             str(compile_dir), makefile_dir, args.cinm_opt, PRE_PASSES, prim)
            for fn, seed, params, _ in tasks
        ]
        print(f"\nCompiling with {args.workers} workers...")
        dpus_for = {(fn, seed): nd for fn, seed, _, nd in tasks}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(compile_one, a): (a[0], a[1]) for a in compile_args}
            for fut in tqdm(as_completed(futures), desc="Compiling", total=len(futures)):
                fn_name, seed = futures[fut]
                _, _, ok, msg = fut.result()
                status = "OK" if ok else "FAIL"
                tqdm.write(f"  [{status}] {fn_name} seed_{seed}")
                if not ok:
                    tqdm.write(f"         {msg}", file=sys.stderr)
                if ok:
                    compiled.append((fn_name, seed, dpus_for[fn_name, seed]))
    else:
        compiled = [
            (fn, seed, nd) for fn, seed, _, nd in tasks
            if not (compile_dir / fn / f"seed_{seed}" / "failed").exists()
        ]

    # ── Benchmark phase ───────────────────────────────────────────────────────
    if not args.compile_only:
        print(f"\nRunning {len(compiled)} benchmarks (DPU cap={args.dpu_cap})...")
        pending   = list(compiled)
        in_flight = {}
        used_dpus = 0
        retry_pending = []

        count_total = len(pending)
        count_done = 0
        run_workers = max(min(count_total, args.workers), 1)
        with ThreadPoolExecutor(max_workers=run_workers) as ex:
            pbar = tqdm(total=count_total, desc="Running")
            while pending or in_flight:
                remaining = []
                for fn_name, seed, num_dpus in pending:
                    if used_dpus + num_dpus <= args.dpu_cap:
                        fut = ex.submit(run_one, fn_name, seed, str(compile_dir), str(run_dir), args.iters)
                        in_flight[fut] = (fn_name, seed, num_dpus)
                        used_dpus += num_dpus
                    else:
                        remaining.append((fn_name, seed, num_dpus))
                pending = remaining

                if not in_flight:
                    fn_name, seed, num_dpus = pending.pop(0)
                    tqdm.write(f"  WARNING: {fn_name} seed_{seed} needs "
                               f"{num_dpus} DPUs > cap {args.dpu_cap}, running alone",
                               file=sys.stderr)
                    fut = ex.submit(run_one, fn_name, seed, str(compile_dir), str(run_dir), args.iters)
                    in_flight[fut] = (fn_name, seed, num_dpus)
                    used_dpus += num_dpus

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    count_done += 1

                    fn_name, seed, num_dpus = in_flight.pop(fut)
                    used_dpus -= num_dpus
                    _, _, ok, msg = fut.result()
                    leader = f"{count_done:>5} / {count_total}"
                    if not ok and is_dpu_allocation_error(msg):
                        tqdm.write(f"[{leader}] [RETRY] {fn_name} seed_{seed} hit a DPU allocation error")
                        retry_pending.append((fn_name, seed, num_dpus))
                        continue
                    status = "OK" if ok else "FAIL"
                    tqdm.write(f"[{leader}]  [{status}] {fn_name} seed_{seed}"
                               f"  (used={used_dpus}/{args.dpu_cap} DPUs)")
                    if not ok:
                        tqdm.write(f"         {msg[:300]}", file=sys.stderr)
                    pbar.update(1)

            if retry_pending:
                tqdm.write(f"\nRetrying {len(retry_pending)} config(s) sequentially...")
                for fn_name, seed, num_dpus in retry_pending:
                    fut = ex.submit(run_one, fn_name, seed, str(compile_dir), str(run_dir), args.iters)
                    _, _, ok, msg = fut.result()
                    status = "OK" if ok else "FAIL"
                    tqdm.write(f"  [{status}] {fn_name} seed_{seed}  (retry)")
                    if not ok:
                        tqdm.write(f"         {msg[:300]}", file=sys.stderr)
                    pbar.update(1)

            pbar.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
