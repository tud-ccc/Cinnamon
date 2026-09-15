"""Drive cinm-opt's accelerator search over a module, and look at what it did.

Four commands produce a dump directory, in rising order of cost: `space`
builds each function's configuration space and evaluates nothing, `sample`
prices a uniform draw from it, `search` runs the Bayesian optimiser, and
`exhaustive` prices every feasible configuration. Four more read one back:
`plot` for search quality, `diag` for what the search was doing per round,
`profiles` for the graph allocator's per-class curves, `analyze` for the
shape of the landscape, and `view` for the interactive pool browser.

The cinm-opt invocations are cinm_experiments.cinmopt's, so this and the
experiment pipelines drive the compiler through one set of wrappers rather
than each assembling --upmem-infer-accelerator strings of its own.

    python -m cinm_bo space  benchmarks/prim/gemv.mlir out/
    python -m cinm_bo search benchmarks/prim/gemv.mlir out/ --seeds 8
    python -m cinm_bo plot   out/ --out-dir out/plots
    python -m cinm_bo diag   out/ -o out/diag
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from cinm_experiments import cinmopt


# ── helpers ───────────────────────────────────────────────────────────────────


def _opt_pair(text: str) -> tuple[str, str]:
    """A `key=value` pass option, as accepted by --upmem-infer-accelerator."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"expected key=value, got {text!r} "
            "(a flag-like option is key=true, e.g. -O dump-full-pool=true)"
        )
    key, value = text.split("=", 1)
    return key, value


def _infer_opts(args: argparse.Namespace) -> dict:
    return dict(args.opt or [])


def _module_cmd(module: str, *rest: str) -> list[str]:
    """Run a sibling module as its own process.

    The dump readers are separate programs with their own option sets, and
    this forwards the caller's extra arguments to them verbatim rather than
    re-declaring them. `-m` rather than a file path: they are modules of this
    package, wherever it happens to be installed.
    """
    return [sys.executable, "-m", f"cinm_bo.{module}", *rest]


def _pool_csvs(dump: pathlib.Path) -> list[pathlib.Path]:
    """Every pool.csv under a dump directory.

    A multi-seed search writes infer_<fn>/seed_<k>/pool.csv; `sample` and
    `exhaustive` write infer_<fn>/pool.csv. Both are worth plotting, so look
    for the seeded layout and fall back to the flat one.
    """
    seeded = sorted(dump.glob("*/seed_*/pool.csv"))
    return seeded or sorted(dump.glob("*/pool.csv"))


def _run_dirs(dump: pathlib.Path) -> list[pathlib.Path]:
    """The directories holding per-round diagnostics (rounds.csv)."""
    return sorted(p.parent for p in dump.glob("**/rounds.csv"))


# ── commands that run cinm-opt ────────────────────────────────────────────────


def cmd_space(args: argparse.Namespace) -> int:
    cinmopt.dump_space(
        args.src, args.out, infer_opts=_infer_opts(args), cinm_opt=args.cinm_opt
    )
    print(f"[space] {args.out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cinmopt.bo_multiseed(
        args.src,
        args.out,
        n_seeds=args.seeds,
        offset=args.offset,
        workers=args.workers,
        infer_opts=_infer_opts(args),
        cinm_opt=args.cinm_opt,
        debug=args.debug,
    )
    print(f"[search] {args.seeds} seed(s) -> {args.out}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    cinmopt.random_sample(
        args.src,
        args.out,
        n_samples=args.n,
        seed=args.seed,
        workers=args.workers,
        infer_opts=_infer_opts(args),
        cinm_opt=args.cinm_opt,
    )
    print(f"[sample] {args.n} configs -> {args.out}")
    return 0


def cmd_exhaustive(args: argparse.Namespace) -> int:
    cinmopt.exhaustive_search(
        args.src,
        args.out,
        workers=args.workers,
        infer_opts=_infer_opts(args),
        cinm_opt=args.cinm_opt,
    )
    print(f"[exhaustive] {args.out}")
    return 0


# ── commands that read a dump ─────────────────────────────────────────────────


def cmd_plot(args: argparse.Namespace) -> int:
    """Search-quality plots over a dump directory (cinm_bo.plot_bo)."""
    pools = _pool_csvs(args.dump)
    if not pools:
        print(f"[plot] no pool.csv under {args.dump}", file=sys.stderr)
        return 1

    plot_args: list[str] = []
    if args.oracle:
        # plot_bo pairs each --oracle CSV with the seed CSVs that follow it,
        # matched by the problem directory's name.
        for oracle_dir in sorted(p for p in args.oracle.iterdir() if p.is_dir()):
            oracle_csv = oracle_dir / "pool.csv"
            seed_csvs = sorted((args.dump / oracle_dir.name).glob("seed_*/pool.csv"))
            if oracle_csv.exists() and seed_csvs:
                plot_args += ["--oracle", str(oracle_csv)]
                plot_args += [str(p) for p in seed_csvs]
        if not plot_args:
            print(
                f"[plot] no problem in {args.dump} matches {args.oracle}",
                file=sys.stderr,
            )
            return 1
    else:
        plot_args = [str(p) for p in pools]

    return subprocess.run(
        _module_cmd(
            "plot_bo",
            "--out-dir",
            str(args.out_dir),
            "--objective-scale",
            args.scale,
            *(["--no-per-seed"] if args.no_per_seed else []),
            *plot_args,
            *args.rest,
        )
    ).returncode


def cmd_diag(args: argparse.Namespace) -> int:
    """Per-round search diagnostics (cinm_bo.plot_search_diag)."""
    rundirs = _run_dirs(args.dump)
    if not rundirs:
        print(
            f"[diag] no rounds.csv under {args.dump} -- per-round diagnostics "
            "are only written by a search, not by sample or exhaustive",
            file=sys.stderr,
        )
        return 1
    return subprocess.run(
        _module_cmd(
            "plot_search_diag",
            "-o",
            str(args.out_dir),
            *[str(d) for d in rundirs],
            *args.rest,
        )
    ).returncode


def cmd_profiles(args: argparse.Namespace) -> int:
    """Per-class cost profiles and what the allocator did with them
    (cinm_bo.plot_profiles)."""
    if not any(args.dump.rglob("profiles.csv")):
        print(
            f"[profiles] no profiles.csv under {args.dump} -- these are the "
            "graph allocator's dumps, written with graph-allocation=true",
            file=sys.stderr,
        )
        return 1
    return subprocess.run(
        _module_cmd(
            "plot_profiles", str(args.dump), "--out", str(args.out_dir), *args.rest
        )
    ).returncode


def cmd_analyze(args: argparse.Namespace) -> int:
    """Landscape analysis (cinm_bo.analyze_landscape)."""
    return subprocess.run(
        _module_cmd(
            "analyze_landscape",
            "--in-dir",
            str(args.dump),
            "--out-dir",
            str(args.out_dir),
            *args.rest,
        )
    ).returncode


def cmd_view(args: argparse.Namespace) -> int:
    """Interactive pool browser (cinm_bo.view_pool)."""
    return subprocess.run(
        _module_cmd("view_pool", str(args.csv), "--scale", args.scale, *args.rest)
    ).returncode


# ── argument parsing ──────────────────────────────────────────────────────────


def _add_compile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("src", type=pathlib.Path, help="the module to search over")
    p.add_argument("out", type=pathlib.Path, help="dump directory to write")
    p.add_argument(
        "-O",
        "--opt",
        type=_opt_pair,
        action="append",
        metavar="KEY=VALUE",
        help="an --upmem-infer-accelerator option; repeatable",
    )
    # No default: None reaches cinmopt, which resolves whichever build the
    # environment holds. Resolving it here would need an installed compiler
    # just to print --help.
    p.add_argument(
        "--cinm-opt",
        type=pathlib.Path,
        default=None,
        help="the cinm-opt to drive (default: the environment's)",
    )


def _add_reader_args(p: argparse.ArgumentParser, *, out_default: str) -> None:
    p.add_argument("dump", type=pathlib.Path, help="a dump directory to read")
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help=f"where to write (default: <dump>/{out_default})",
    )
    # Anything this parser does not know is forwarded to the module the
    # command drives (see main); declaring it as a REMAINDER positional
    # instead would swallow --out-dir whenever it followed the dump.


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m cinm_bo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p = sub.add_parser("space", help="dump each function's config space")
    _add_compile_args(p)
    p.set_defaults(func=cmd_space)

    p = sub.add_parser("search", help="run the Bayesian search")
    _add_compile_args(p)
    p.add_argument("--seeds", type=int, default=1, help="independent searches")
    p.add_argument(
        "--offset",
        type=int,
        default=67,
        help="rng offset; seed k uses offset + k*31",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="-debug-only=cinm-inference")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("sample", help="price a uniform draw of feasible configs")
    _add_compile_args(p)
    p.add_argument("-n", type=int, required=True, help="configs to draw")
    p.add_argument("--seed", type=int, default=None, help="draw seed")
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("exhaustive", help="price every feasible config")
    _add_compile_args(p)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_exhaustive)

    p = sub.add_parser("plot", help="search-quality plots")
    _add_reader_args(p, out_default="plots")
    p.add_argument(
        "--oracle",
        type=pathlib.Path,
        help="a dump directory of ground truth to compare against",
    )
    p.add_argument("--scale", default="log10")
    p.add_argument("--no-per-seed", action="store_true")
    p.set_defaults(func=cmd_plot, out_default="plots")

    p = sub.add_parser("diag", help="per-round search diagnostics")
    _add_reader_args(p, out_default="diag")
    p.set_defaults(func=cmd_diag, out_default="diag")

    p = sub.add_parser("profiles", help="per-class cost profiles and allocation")
    _add_reader_args(p, out_default="profiles")
    p.set_defaults(func=cmd_profiles, out_default="profiles")

    p = sub.add_parser("analyze", help="landscape analysis of a pool")
    _add_reader_args(p, out_default="landscape")
    p.set_defaults(func=cmd_analyze, out_default="landscape")

    p = sub.add_parser("view", help="interactive pool browser")
    p.add_argument("csv", type=pathlib.Path, help="a pool.csv")
    p.add_argument("--scale", default="log10")
    p.set_defaults(func=cmd_view)

    return root


def main(argv: list[str] | None = None) -> int:
    # Options this CLI does not define belong to the module the command
    # drives -- plot_bo's --axes, plot_search_diag's --q -- and are handed
    # over untouched rather than redeclared here.
    args, rest = build_parser().parse_known_args(argv)
    args.rest = rest
    if getattr(args, "out_dir", None) is None and hasattr(args, "out_default"):
        args.out_dir = args.dump / args.out_default
    try:
        return args.func(args) or 0
    except RuntimeError as exc:
        # What cinmopt raises when cinm-opt exits non-zero; its message names
        # the log file, which is where the actual error is.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
