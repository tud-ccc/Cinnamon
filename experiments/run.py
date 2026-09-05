#!/usr/bin/env python3
"""
Experiment runner for cinm-mlir UPMEM inference experiments.

Subcommands
-----------
  run          Run cinm-opt for a single seed, then plot.
  seeds        Run cinm-opt for N seeds in parallel (ProcessPoolExecutor), then plot.
  exhaustive   Run cinm-opt with exhaustive search (oracle / ground truth).
  plot         Plot an already-populated data directory.
  view         Interactive pool viewer.
  analyze      Landscape analysis.

Common pattern:
  python run.py run gemv --scale log10
  python run.py seeds gemv --n 10 --workers 4 --oracle data/gemv_oracle
  python run.py exhaustive gemv
  python run.py plot gemv --oracle data/gemv_oracle -- --objective-scale log10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = EXPERIMENTS_DIR.parent / ".venv" / "bin" / "python"


def python_bin() -> str:
    """Return the venv python if it exists, otherwise the current interpreter."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


# ── cinm-opt invocation helpers ────────────────────────────────────────────────


def _infer_opts(
    *,
    scale: str,
    dump_dir: str,
    seed: int,
    extra: list[str],
) -> str:
    parts = list(extra) + [
        f"objective-scale={scale}",
        f"dump-dir={dump_dir}",
        f"rng-seed={seed}",
    ]
    return " ".join(parts)


def _cinm_opt_cmd(
    file: str,
    infer_opts: str,
    *,
    cinm_opt: str = "cinm-opt",
    extra_mlir_flags: list[str] | None = None,
    out_file=None,
) -> list[str]:
    cmd = [
        cinm_opt,
        str(file),
        "--cinm-assign-platforms",
        "--cinm-isolate-compute-blocks",
        f"--upmem-infer-accelerator={infer_opts}",
        "--mlir-print-ir-after-failure",
        "--dump-pass-pipeline",
        "--split-input-file",
        "--mlir-disable-threading",
        "--debug-only=cinm-inference",
    ]
    if out_file:
        cmd.extend(("-o", str(out_file)))
    if extra_mlir_flags:
        cmd.extend(extra_mlir_flags)
    # print(" ".join(cmd))
    return cmd


# ── seed worker (top-level so ProcessPoolExecutor can pickle it) ───────────────


def _run_seed(args: tuple) -> tuple[int, int]:
    """Run one cinm-opt seed. Returns (seed, returncode)."""
    seed, file, dump_dir, scale, extra, cinm_opt = args
    infer_opts = _infer_opts(scale=scale, dump_dir=dump_dir, seed=seed, extra=extra)
    cmd = _cinm_opt_cmd(file, infer_opts, cinm_opt=cinm_opt)
    log_path = Path(dump_dir) / f"seed{seed}.log"
    out_path = Path(dump_dir) / f"seed{seed}_out.mlir"
    with open(log_path, "w") as log_f, open(out_path, "w") as out_f:
        result = subprocess.run(cmd, stderr=log_f, stdout=out_f)
    return seed, result.returncode


# ── subcommands ────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    """Single-seed run followed by plot."""
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    infer_opts = _infer_opts(
        scale=args.scale,
        dump_dir=str(data_dir),
        seed=args.seed,
        extra=args.extra,
    )
    cmd = _cinm_opt_cmd(args.file, infer_opts, cinm_opt=args.cinm_opt)

    log_path = data_dir / f"{args.file}_seed{args.seed}.log"
    out_path = data_dir / "out.mlir"

    print(f"[run] seed: {args.seed}  log: {log_path}")
    with open(log_path, "w") as log_f, open(out_path, "w") as out_f:
        rc = subprocess.run(cmd, stderr=log_f, stdout=out_f).returncode

    if rc != 0:
        print(f"[run] cinm-opt exited {rc} — see {log_path}", file=sys.stderr)

    args.in_dir = args.out_dir
    return cmd_plot(args) or rc


def cmd_seeds(args: argparse.Namespace) -> int:
    """Run N seeds in parallel, then plot."""
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = [i * 31 + args.offset for i in range(1, args.n + 1)]
    worker_args = [
        (seed, args.file, str(data_dir), args.scale, args.extra, args.cinm_opt)
        for seed in seeds
    ]

    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"[seeds] {args.n} seeds, {workers} workers, dir={args.out_dir}")

    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_seed, wa): wa[0] for wa in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            seed = futures[fut]
            try:
                _, rc = fut.result()
            except Exception as exc:
                print(f"[seeds] seed: {seed} raised: {exc}", file=sys.stderr)
                failed += 1
                continue
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            tqdm.write(f"[seeds] seed: {seed}  {status}")
            if rc != 0:
                failed += 1

    if failed:
        print(f"[seeds] WARNING: {failed}/{args.n} seed(s) failed", file=sys.stderr)

    code = 1 if failed else 0
    if args.no_plots:
        return code
    args.in_dir = args.out_dir
    return cmd_plot(args)


def cmd_multiseed(args: argparse.Namespace) -> int:
    """Run N seeds in a single cinm-opt process (C++ multi-seed engine), then
    plot. Unlike `seeds` (process-per-seed), this shares the config space, the
    valid-config scan, and the validation set across seeds and runs them
    concurrently in-process. Produces the same seed_<value>/ dump layout."""
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extra = list(args.extra) + [f"n-seeds={args.n}"]
    if args.workers:
        extra.append(f"n-workers={args.workers}")
    # rng-seed acts as the seed offset: seed k (1..n) uses k*31 + offset, matching
    # the `seeds` command's scheme so the two engines produce comparable seeds.
    infer_opts = _infer_opts(
        scale=args.scale, dump_dir=str(data_dir), seed=args.offset, extra=extra
    )
    out_path = data_dir / "out.mlir"
    cmd = _cinm_opt_cmd(
        args.file, infer_opts, cinm_opt=args.cinm_opt, out_file=out_path
    )

    print(
        f"[multiseed] {args.n} seeds, workers={args.workers or 'auto'}, "
        f"dir={args.out_dir}"
    )

    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[multiseed] cinm-opt exited {rc}", file=sys.stderr)

    if args.no_plots:
        return rc
    args.in_dir = args.out_dir
    return cmd_plot(args) or rc


def cmd_exhaustive(args: argparse.Namespace) -> int:
    """Run exhaustive search (oracle / ground truth)."""
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    infer_opts = f"dump-dir={data_dir} exhaustive-search {' '.join(args.extra)}"
    cmd = _cinm_opt_cmd(args.file, infer_opts, cinm_opt=args.cinm_opt)

    log_path = data_dir / f"{args.file}.log"
    out_path = data_dir / "out.mlir"

    print(f"[exhaustive] log={log_path}")
    with open(out_path, "w") as out_f:
        return subprocess.run(cmd, stdout=out_f).returncode


def cmd_plot(args: argparse.Namespace) -> int:
    """Call plot_bo.py on the data directory."""
    in_dir = Path(args.in_dir)
    out_dir = str(args.out_dir)
    oracle = getattr(args, "oracle", None) or ""
    scale = getattr(args, "scale", "log10")
    extra = getattr(args, "plot_extra", [])
    no_per_seed = getattr(args, "no_per_seed", False)
    plots_filter = getattr(args, "plots", None) or []

    plot_script = EXPERIMENTS_DIR / "plotting" / "plot_bo.py"

    if oracle:
        # Build --oracle <oracle_subdir/pool.csv> <seed_csvs...> pairs per subdir
        oracle_path = Path(oracle)
        plot_args: list[str] = []
        for subdir in sorted(oracle_path.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name
            oracle_csv = subdir / "pool.csv"
            seed_csvs = sorted((in_dir / name).glob("seed_*/pool.csv"))
            if oracle_csv.exists() and seed_csvs:
                plot_args += ["--oracle", str(oracle_csv)]
                plot_args += [str(p) for p in seed_csvs]
    else:
        plot_args = [str(p) for p in sorted(in_dir.glob("*/seed_*/pool.csv"))]

    cmd = [
        python_bin(),
        str(plot_script),
        "--objective-scale",
        scale,
        "--out-dir",
        out_dir,
        *(["--no-per-seed"] if no_per_seed else []),
        *plot_args,
        *extra,
        *(["--plots", *plots_filter] if plots_filter else []),
    ]
    code = subprocess.run(cmd).returncode
    if code != 0:
        print("FAILED" + " ".join(cmd))
    return code


def cmd_view(args: argparse.Namespace) -> int:
    """Launch the interactive pool viewer."""
    pool_csv = Path("data") / args.name / "pool.csv"
    cmd = [
        python_bin(),
        str(EXPERIMENTS_DIR / "viewer" / "view_pool.py"),
        str(pool_csv),
        "--scale",
        args.scale,
    ]
    return subprocess.run(cmd).returncode


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run landscape analysis (single problem dir or parent dir of many problems)."""
    cmd = [
        python_bin(),
        str(EXPERIMENTS_DIR / "plotting" / "analyze_landscape.py"),
        "--in-dir",
        args.in_dir,
        "--out-dir",
        args.out_dir,
        *args.extra,
    ]
    return subprocess.run(cmd).returncode


# ── Argument parsing ───────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    """Add args shared by run / seeds / exhaustive."""
    p.add_argument(
        "file", metavar="FILE", help="Input file stem (without .mlir extension)"
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        required=True,
        metavar="DIR",
        help="Output directory",
    )
    p.add_argument(
        "--cinm-opt", default="cinm-opt", metavar="PATH", help="Path to cinm-opt binary"
    )


def _add_scale(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--scale",
        default="log10",
        choices=["linear", "log2", "log10", "ln", "sqrt", "cbrt"],
        help="Objective scale (default: log10)",
    )


def _add_oracle(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--oracle",
        metavar="DIR",
        default="",
        help="Path to exhaustive-search data dir for plot overlay",
    )


def _add_extra(
    p: argparse.ArgumentParser,
    dest: str = "extra",
    help: str = "Extra options forwarded to --upmem-infer-accelerator",
) -> None:
    p.add_argument(
        "--infer-opts",
        dest="extra",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help=help,
    )


def _add_plot_extra(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--",
        dest="plot_extra",
        nargs="*",
        default=[],
        metavar="ARG",
        help="Extra args forwarded to plot_bo.py",
    )


def _add_plots_filter(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--plots",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only generate plots whose tag contains one of these substrings "
        "(forwarded to plot_bo.py --plots)",
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="cmd", metavar="SUBCOMMAND", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Single-seed run + plot")
    _add_common(p_run)
    _add_scale(p_run)
    _add_oracle(p_run)
    p_run.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p_run.add_argument(
        "--no-per-seed",
        action="store_true",
        dest="no_per_seed",
        help="Pass --no-per-seed to plot_bo.py",
    )
    _add_plots_filter(p_run)
    _add_extra(p_run)
    p_run.set_defaults(func=cmd_run)

    # ── seeds ────────────────────────────────────────────────────────────────
    p_seeds = sub.add_parser(
        "seeds", help="Run N seeds in parallel (ProcessPoolExecutor) + plot"
    )
    _add_common(p_seeds)
    _add_scale(p_seeds)
    _add_oracle(p_seeds)
    p_seeds.add_argument(
        "-n", "--n", type=int, default=5, help="Number of seeds (default: 5)"
    )
    p_seeds.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: ncpu-2)",
    )
    p_seeds.add_argument(
        "--offset",
        type=int,
        default=67,
        help="Offset to use to make generated seeds different from another run of the command",
    )
    p_seeds.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Don't run the plotting code",
    )
    p_seeds.add_argument(
        "--no-per-seed",
        action="store_true",
        dest="no_per_seed",
        help="Pass --no-per-seed to plot_bo.py",
    )
    _add_plots_filter(p_seeds)
    _add_extra(p_seeds)
    p_seeds.set_defaults(func=cmd_seeds)

    # ── multiseed ────────────────────────────────────────────────────────────
    p_ms = sub.add_parser(
        "multiseed",
        help="Run N seeds concurrently in one cinm-opt process (shared init) + plot",
    )
    _add_common(p_ms)
    _add_scale(p_ms)
    _add_oracle(p_ms)
    p_ms.add_argument(
        "-n", "--n", type=int, default=5, help="Number of seeds (default: 5)"
    )
    p_ms.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Max concurrent seeds / n-workers (default: auto = hw threads)",
    )
    p_ms.add_argument(
        "--offset",
        type=int,
        default=67,
        help="Seed offset (seed k uses k*31 + offset); forwarded as rng-seed",
    )
    p_ms.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Don't run the plotting code",
    )
    p_ms.add_argument(
        "--no-per-seed",
        action="store_true",
        dest="no_per_seed",
        help="Pass --no-per-seed to plot_bo.py",
    )
    _add_plots_filter(p_ms)
    _add_extra(p_ms)
    p_ms.set_defaults(func=cmd_multiseed)

    # ── exhaustive ───────────────────────────────────────────────────────────
    p_ex = sub.add_parser("exhaustive", help="Exhaustive search (oracle)")
    _add_common(p_ex)
    _add_extra(p_ex)
    p_ex.set_defaults(func=cmd_exhaustive)

    # ── plot ─────────────────────────────────────────────────────────────────
    p_plot = sub.add_parser("plot", help="Plot an existing data directory")
    p_plot.add_argument(
        "--in-dir",
        dest="in_dir",
        required=True,
        metavar="DIR",
        help="Input directory containing pool.csv files",
    )
    p_plot.add_argument(
        "--out-dir",
        dest="out_dir",
        required=True,
        metavar="DIR",
        help="Output directory for plots",
    )
    _add_scale(p_plot)
    _add_oracle(p_plot)
    p_plot.add_argument(
        "--no-per-seed",
        action="store_true",
        dest="no_per_seed",
        help="Pass --no-per-seed to plot_bo.py",
    )
    _add_plots_filter(p_plot)
    p_plot.add_argument(
        "plot_extra",
        nargs="*",
        metavar="ARG",
        help="Extra args forwarded to plot_bo.py",
    )
    p_plot.set_defaults(func=cmd_plot, file=None)

    # ── view ─────────────────────────────────────────────────────────────────
    p_view = sub.add_parser("view", help="Interactive pool viewer")
    p_view.add_argument("name", metavar="NAME", help="Subdirectory under data/")
    _add_scale(p_view)
    p_view.set_defaults(func=cmd_view)

    # ── analyze ──────────────────────────────────────────────────────────────
    p_an = sub.add_parser("analyze", help="Landscape analysis")
    p_an.add_argument(
        "--in-dir",
        dest="in_dir",
        required=True,
        metavar="DIR",
        help="Input: pool.csv, problem dir, or parent dir of problems",
    )
    p_an.add_argument(
        "--out-dir",
        dest="out_dir",
        required=True,
        metavar="DIR",
        help="Output directory for analysis plots and README",
    )
    _add_extra(p_an, help="Extra args forwarded to analyze_landscape.py")
    p_an.set_defaults(func=cmd_analyze)

    return root


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
