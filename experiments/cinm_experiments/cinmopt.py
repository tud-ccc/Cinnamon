"""Python wrappers around cinm-opt --upmem-infer-accelerator invocations.

These are thin subprocess wrappers -- cinm-opt is a C++ binary, so it can't be
called in-process -- but they turn "build a shell command line" into ordinary
function calls with real arguments, so experiment scripts never construct
--upmem-infer-accelerator option strings by hand.
"""
from __future__ import annotations

import pathlib
import subprocess

from .paths import DEFAULT_CINM_OPT

PRE_PASSES = ["--cinm-assign-platforms", "--cinm-isolate-compute-blocks"]


def _opt_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _infer_opts_str(opts: dict) -> str:
    return " ".join(f"{k}={_opt_value(v)}" for k, v in opts.items())


def _run(src: pathlib.Path, infer_opts: dict, *, out_file: pathlib.Path,
          cinm_opt: pathlib.Path, log_file: pathlib.Path,
          nice: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        str(cinm_opt), str(src),
        "--split-input-file",
        *PRE_PASSES,
        f"--upmem-infer-accelerator={_infer_opts_str(infer_opts)}",
        "-o", str(out_file),
    ]
    if nice:
        cmd = ["nice", "-n", "19", *cmd]
    with open(log_file, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


def exhaustive_search(src: pathlib.Path, out_dir: pathlib.Path, *, workers: int | None = None,
                       infer_opts: dict | None = None, nice: bool = True,
                       cinm_opt: pathlib.Path = DEFAULT_CINM_OPT) -> pathlib.Path:
    """Exhaustively evaluate every valid config, dumping
    {out_dir}/infer_{fn_name}/pool.csv per function found in src (the
    "infer_" prefix comes from the pass's own NameInventor; dump-full-pool is
    forced on, since screening needs the full pool, not just visited rows).
    Runs niced (nice -n 19) by default -- exhaustive search is CPU-hungry and
    this is usually run alongside other work. Returns out_dir."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"dump-dir": str(out_dir), "exhaustive-search": True,
            "dump-full-pool": True, **({"n-workers": workers} if workers else {}),
            **(infer_opts or {})}
    r = _run(src, opts, out_file=out_dir / "out.mlir", cinm_opt=cinm_opt,
             log_file=out_dir / "cinm-opt.log", nice=nice)
    if r.returncode != 0:
        raise RuntimeError(f"exhaustive_search failed for {src}; see {out_dir}/cinm-opt.log")
    return out_dir


def bo_multiseed(src: pathlib.Path, out_dir: pathlib.Path, *, n_seeds: int = 32,
                  offset: int = 67, workers: int | None = None,
                  infer_opts: dict | None = None, nice: bool = False,
                  cinm_opt: pathlib.Path = DEFAULT_CINM_OPT) -> pathlib.Path:
    """Run n_seeds independent BO searches sharing the config-space setup
    (the C++ multi-seed engine), dumping
    {out_dir}/infer_{fn_name}/seed_<k>/pool.csv for k = 1..n_seeds (seed k
    uses rng offset+k*31; "infer_" prefix as in exhaustive_search()). Returns
    out_dir."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"dump-dir": str(out_dir), "rng-seed": offset, "n-seeds": n_seeds,
            **({"n-workers": workers} if workers else {}), **(infer_opts or {})}
    r = _run(src, opts, out_file=out_dir / "out.mlir", cinm_opt=cinm_opt,
             log_file=out_dir / "cinm-opt.log", nice=nice)
    if r.returncode != 0:
        raise RuntimeError(f"bo_multiseed failed for {src}; see {out_dir}/cinm-opt.log")
    return out_dir


def eval_solution(src: pathlib.Path, params: dict, *, out_file: pathlib.Path,
                   cinm_opt: pathlib.Path = DEFAULT_CINM_OPT, nice: bool = False,
                   log_file: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    """Compile exactly one configuration, no search. `params` must be ordered
    to match the dimensions UpmemInferAccelerator declared for this op (dpus,
    tasklets, then whatever the op handler added) -- see pools.param_cols for
    a way to recover that order from an existing pool.csv."""
    solution_str = ",".join(str(v) for v in params.values())
    log_file = log_file or (pathlib.Path(out_file).parent / "cinm-opt.log")
    return _run(src, {"eval-solution": solution_str}, out_file=out_file,
                cinm_opt=cinm_opt, log_file=log_file, nice=nice)


def eval_solution_lowerer(params: dict, *, cinm_opt: pathlib.Path = DEFAULT_CINM_OPT):
    """A (fn_module, out_file, log_file) -> CompletedProcess callable, for use
    as compile_run.Config.lower -- compiles CINM 2.0's chosen `params` with no
    further search."""

    def _lower(fn_module: pathlib.Path, out_file: pathlib.Path,
               log_file: pathlib.Path) -> subprocess.CompletedProcess:
        return eval_solution(fn_module, params, out_file=out_file,
                              cinm_opt=cinm_opt, log_file=log_file)

    return _lower
