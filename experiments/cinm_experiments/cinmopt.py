"""Python wrappers around cinm-opt --upmem-infer-accelerator invocations.

These are thin subprocess wrappers -- cinm-opt is a C++ binary, so it can't be
called in-process -- but they turn "build a shell command line" into ordinary
function calls with real arguments, so experiment scripts never construct
--upmem-infer-accelerator option strings by hand.
"""

from __future__ import annotations

import pathlib
import subprocess
import shlex

from .paths import DEFAULT_CINM_OPT

PRE_PASSES = ["--cinm-assign-platforms", "--cinm-isolate-compute-blocks"]


def _opt_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _infer_opts_str(opts: dict) -> str:
    return " ".join(f"{k}={_opt_value(v)}" for k, v in opts.items())


def _run(
    src: pathlib.Path,
    infer_opts: dict,
    *,
    out_file: pathlib.Path,
    cinm_opt: pathlib.Path,
    log_file: pathlib.Path,
    extra_opts: list = [],
    pre_passes: list = PRE_PASSES,
    nice: bool = False,
    nolog: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [
        str(cinm_opt),
        str(src),
        "--split-input-file",
        *pre_passes,
        "--debug-only=cinm-inference",
        f"--upmem-infer-accelerator={_infer_opts_str(infer_opts)}",
        *extra_opts,
        "-o",
        str(out_file),
    ]
    if nice:
        cmd = ["nice", "-n", "1", *cmd]
    with open(log_file, "w") as log:
        log.write(shlex.join(cmd) + "\n\n")
        # subprocess.run(stdout=log) hands the child the fd directly, which
        # writes straight to the OS file at its current position -- without
        # this flush, the write() above is still sitting in Python's
        # userspace buffer (not yet at that position), so the child's output
        # lands first and the echoed command line gets appended after it
        # once the buffer finally flushes at file-close time, silently
        # reordering the log.
        log.flush()

        if nolog:
            return subprocess.run(cmd)
        else:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


def exhaustive_search(
    src: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    workers: int | None = None,
    infer_opts: dict | None = None,
    nice: bool = True,
    out_file: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> pathlib.Path:
    """Exhaustively evaluate every valid config, dumping
    {out_dir}/infer_{fn_name}/pool.csv per function found in src (the
    "infer_" prefix comes from the pass's own NameInventor; dump-full-pool is
    forced on, since screening needs the full pool, not just visited rows).
    Runs niced (nice -n 19) by default -- exhaustive search is CPU-hungry and
    this is usually run alongside other work. Returns out_dir.

    See bo_multiseed for out_file/log_file."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_file or out_dir / "out.mlir"
    log_file = log_file or out_dir / "cinm-opt.log"
    opts = {
        "dump-dir": str(out_dir),
        "exhaustive-search": True,
        "dump-full-pool": True,
        **({"n-workers": workers} if workers else {}),
        **(infer_opts or {}),
    }
    r = _run(
        src,
        opts,
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
        nice=nice,
    )
    if r.returncode != 0:
        raise RuntimeError(f"exhaustive_search failed for {src}; see {log_file}")
    return out_dir


def dump_space(
    src: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    infer_opts: dict | None = None,
    out_file: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> pathlib.Path:
    """Build each function's config space and dump
    {out_dir}/infer_{fn_name}/space.json, evaluating nothing and committing
    nothing (the pass's dump-space-only mode). The dump carries per-param
    doc strings and, for permutation params, the full orderings table with
    copy-pasteable eval-solution assignments -- the input of the manual
    ATiM-transcription workflow. Cheap: no simulator runs, only the
    constraint solve. Returns out_dir.

    See bo_multiseed for out_file/log_file."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_file or out_dir / "out.mlir"
    log_file = log_file or out_dir / "cinm-opt.log"
    opts = {
        "dump-dir": str(out_dir),
        "dump-space-only": True,
        **(infer_opts or {}),
    }
    r = _run(
        src,
        opts,
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
    )
    if r.returncode != 0:
        raise RuntimeError(f"dump_space failed for {src}; see {log_file}")
    return out_dir


def random_sample(
    src: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    n_samples: int,
    workers: int | None = None,
    infer_opts: dict | None = None,
    seed: int | None = None,
    nice: bool = True,
    out_file: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> pathlib.Path:
    """Evaluate a random sample of n_samples valid configs (instead of every
    valid config, see exhaustive_search), dumping {out_dir}/infer_{fn_name}/
    pool.csv per function found in src. Cheap alternative to
    exhaustive_search when only a small ground-truth sample is needed --
    exhaustive search's cost is entirely the O(n_valid) simulator calls (not
    the O(N) validity scan), so this is O(n_samples) instead, turning
    hours-long full-space sweeps into a low-minutes/seconds run. dump-full-pool
    is left off (the InferenceOptions default), so pool.csv only contains the
    n_samples visited rows, not the whole space. Returns out_dir.

    See bo_multiseed for out_file/log_file. The draw itself does not depend on
    how many functions src holds: the pass builds one InferenceTask per
    compute block, each seeding its own RNG from rng-seed, so a per-function
    run reproduces that function's rows exactly."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_file or out_dir / "out.mlir"
    log_file = log_file or out_dir / "cinm-opt.log"
    opts = {
        "dump-dir": str(out_dir),
        "sample-n": n_samples,
        **({"rng-seed": seed} if seed is not None else {}),
        **({"n-workers": workers} if workers else {}),
        **(infer_opts or {}),
    }
    r = _run(
        src,
        infer_opts=opts,
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
        nice=nice,
    )
    if r.returncode != 0:
        raise RuntimeError(f"random_sample failed for {src}; see {log_file}")
    return out_dir


def graph_allocation(
    src: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    workers: int | None = None,
    infer_opts: dict | None = None,
    nice: bool = True,
    out_file: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> pathlib.Path:
    """Run the two-level graph solve over a whole program: profile every
    program-identity class over the device-size menu, then partition the
    device among the classes. Dumps per graph (a connected component of the
    dataflow between compute blocks, named infer_<fn> by the pass's own
    NameInventor) into {out_dir}/infer_<fn>/:

      profiles.csv    one row per (class, menu point) -- the solver's input
      allocation.csv  one summary row -- classes, groups, host/device split,
                      objective
      groups.csv      one row per device set the solve carved out
      class_<i>/      the per-class search dumps (pool.csv, ...)

    Unlike the per-block stacks this searches a whole module rather than a
    split function: the classes of one program are what the device is
    divided between, so they cannot be separated. Returns out_dir.

    The source is expected to carry its compute blocks already (a
    whole-program front end ends with --cinm-complete-compute-graph), so
    only the isolation runs here: re-running --cinm-assign-platforms would
    be a no-op on blocked IR, and leaving it out keeps the host-pinned
    blocks the front end created plainly the front end's business."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_file or out_dir / "out.mlir"
    log_file = log_file or out_dir / "cinm-opt.log"
    opts = {
        "dump-dir": str(out_dir),
        "graph-allocation": True,
        **({"n-workers": workers} if workers else {}),
        **(infer_opts or {}),
    }
    r = _run(
        src,
        opts,
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
        pre_passes=["--cinm-isolate-compute-blocks"],
        nice=nice,
    )
    if r.returncode != 0:
        raise RuntimeError(f"graph_allocation failed for {src}; see {log_file}")
    return out_dir


def bo_multiseed(
    src: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    n_seeds: int = 32,
    offset: int = 67,
    workers: int | None = None,
    infer_opts: dict | None = None,
    nice: bool = False,
    debug: bool = False,
    nolog: bool = False,
    out_file: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> pathlib.Path:
    """Run n_seeds independent BO searches sharing the config-space setup
    (the C++ multi-seed engine), dumping
    {out_dir}/infer_{fn_name}/seed_<k>/pool.csv for k = 1..n_seeds (seed k
    uses rng offset+k*31; "infer_" prefix as in exhaustive_search()). Returns
    out_dir.

    out_file/log_file default to {out_dir}/out.mlir and {out_dir}/cinm-opt.log.
    Naming them explicitly is what lets several single-function searches share
    one dump root: their infer_{fn_name} subtrees never collide, but one
    out.mlir per root would."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_file or out_dir / "out.mlir"
    log_file = log_file or out_dir / "cinm-opt.log"
    opts = {
        "dump-dir": str(out_dir),
        "rng-seed": offset,
        "n-seeds": n_seeds,
        **({"n-workers": workers} if workers else {}),
        **(infer_opts or {}),
    }
    extra_opts = []
    if debug:
        extra_opts = ["--debug-only=cinm-inference"]

    r = _run(
        src,
        opts,
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
        extra_opts=extra_opts,
        nice=nice,
        nolog=nolog,
    )
    if r.returncode != 0:
        raise RuntimeError(f"bo_multiseed failed for {src}; see {log_file}")
    return out_dir


def eval_solution(
    src: pathlib.Path,
    params: dict,
    *,
    out_file: pathlib.Path,
    extra_infer_opts: dict = {},
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
    nice: bool = False,
    log_file: pathlib.Path | None = None,
) -> subprocess.CompletedProcess:
    """Compile exactly one configuration, no search. `params` maps the search
    space's parameter names to values -- e.g. dpus, tasklets, and one
    `op<N>.block<D>` / `op<N>.leaf<D>` per iteration dimension of the Nth
    distributed op. Every parameter the space declares must be present;
    cinm-opt rejects missing or unknown names and lists what it expects."""
    solution_str = ",".join(f"{name}={value}" for name, value in params.items())
    log_file = log_file or (pathlib.Path(out_file).parent / "cinm-opt.log")
    return _run(
        src,
        infer_opts={"eval-solution": solution_str, **extra_infer_opts},
        out_file=out_file,
        cinm_opt=cinm_opt,
        log_file=log_file,
        nice=nice,
    )


# What --upmem-infer-accelerator prints (to the log; _run merges stderr into
# it) when eval-solution-force meets a configuration the space rejects. This
# is the membership verdict itself -- the same space.isEncodable check the
# guarded error uses -- so probing needs no separately materialised pool.
_FORCE_REJECTED_MARKER = "outside the feasible set"


def probe_solution(
    src: pathlib.Path,
    params: dict,
    *,
    log_file: pathlib.Path,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
    nice: bool = True,
) -> str:
    """Force-lower exactly one configuration and classify what happened:

    - "accepted":        in the feasible set, and the lowering succeeded;
    - "accepted_fails":  in the feasible set, but the lowering failed --
                         the paper's accepted-but-fails counter (must be 0);
    - "rejected_lowers": outside the feasible set, yet it lowers fine --
                         a false negative of the constraint system, which is
                         what the rejected-region experiment counts;
    - "rejected_fails":  outside the feasible set, and the lowering agrees.

    Membership comes from the pass's own check (the eval-solution-force
    note in the log), so the verdict cannot drift from what the space
    actually contains. The simulator is op-count: only the lowering's
    verdict matters here, not the cost estimate. The lowered module is
    discarded; the log stays, for failure classification."""
    result = eval_solution(
        src,
        params,
        out_file=pathlib.Path("/dev/null"),
        extra_infer_opts={
            "eval-solution-force": True,
            "simulator": "op-count",
        },
        cinm_opt=cinm_opt,
        nice=nice,
        log_file=log_file,
    )
    rejected = _FORCE_REJECTED_MARKER in log_file.read_text(errors="replace")
    if rejected:
        return "rejected_lowers" if result.returncode == 0 else "rejected_fails"
    return "accepted" if result.returncode == 0 else "accepted_fails"


def search_lowerer(
    *,
    max_evals: int = 64,
    n_init: int | None = None,
    dir_name: str = "search",
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
    extra_infer_opts: dict = {},
):
    """A Config.lower callable that searches instead of evaluating a fixed
    point: the pass runs its own BO over the space and commits its winner,
    with the working group pinned to the config's `dpus` (and `tasklets`, if
    it names one) so that only the tiling is chosen.

    The config's params are the pin here, not the point -- which is what
    makes a config comparable to one at another working-group size: same
    problem, same space, a different resource. The search record (pool.csv,
    space.json) lands in <out_file.parent>/<dir_name>/."""

    def _lower(
        fn_module: pathlib.Path,
        out_file: pathlib.Path,
        log_file: pathlib.Path,
        *,
        dpus: int,
        tasklets: int | None = None,
        **ignored,
    ) -> subprocess.CompletedProcess:
        out_dir = pathlib.Path(out_file).parent / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "fixed-dpus": dpus,
            **({"fixed-tasklets": tasklets} if tasklets is not None else {}),
            "max-evals": max_evals,
            **({"n-init": n_init} if n_init is not None else {}),
            "dump-dir": str(out_dir),
            **extra_infer_opts,
        }
        return _run(
            fn_module,
            opts,
            out_file=out_file,
            cinm_opt=cinm_opt,
            log_file=log_file,
        )

    return _lower


def annotate_costs(
    src: pathlib.Path,
    *,
    log_file: pathlib.Path,
    program_dump_dir: pathlib.Path | None = None,
    costs_csv: pathlib.Path | None = None,
    simulator: str = "cycle-accurate",
    eval_timeout_ms: int | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
) -> subprocess.CompletedProcess:
    """Run --upmem-annotate-costs over an already-lowered module (the "upmem
    dialect" stage Config.lower produces), for its side outputs rather than
    for the annotated IR, which is discarded.

    `program_dump_dir` writes one <kernel>.cnmprog.json per simulated DPU
    program: the symbolic interchange format the Python reference cost model
    reads (third-party/cnm-cost-model/Predictor/cnmprog.py), so the same
    program can be priced by both engines. `costs_csv` is the C++ side's own
    per-category breakdown of that same run -- the number the dumps get
    compared against."""
    opts: dict = {"simulator": simulator}
    if eval_timeout_ms is not None:
        opts["eval-timeout-ms"] = eval_timeout_ms
    if program_dump_dir is not None:
        opts["program-dump-dir"] = str(program_dump_dir)
    if costs_csv is not None:
        opts["costs-csv"] = str(costs_csv)
    cmd = [
        str(cinm_opt),
        str(src),
        f"--upmem-annotate-costs={_infer_opts_str(opts)}",
        "-o",
        "/dev/null",
    ]
    with open(log_file, "w") as log:
        log.write(shlex.join(cmd) + "\n\n")
        log.flush()  # see _run: the child writes to the fd directly
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


def with_program_dump(
    lower,
    *,
    dir_name: str = "cnmprog",
    simulator: str = "cycle-accurate",
    eval_timeout_ms: int | None = None,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
):
    """Wrap a Config.lower callable so that a successful lowering is followed
    by an annotate_costs run dumping each DPU program to
    <out_file.parent>/<dir_name>/, next to a cost.csv of the same run.

    A second cinm-opt invocation rather than an option on the lowering itself:
    the search's simulator runs on every candidate configuration, and only the
    committed one is worth dumping. A failure of the dump is reported but does
    not fail the lowering -- the dumps are a cross-check of the cost model,
    not an input to the compile that follows."""

    def _lower(
        fn_module: pathlib.Path,
        out_file: pathlib.Path,
        log_file: pathlib.Path,
        **params,
    ) -> subprocess.CompletedProcess:
        r = lower(fn_module, out_file, log_file, **params)
        if r.returncode != 0:
            return r
        dump_dir = pathlib.Path(out_file).parent / dir_name
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump = annotate_costs(
            out_file,
            log_file=dump_dir / "cinm-opt.log",
            program_dump_dir=dump_dir,
            costs_csv=dump_dir / "cost.csv",
            simulator=simulator,
            eval_timeout_ms=eval_timeout_ms,
            cinm_opt=cinm_opt,
        )
        if dump.returncode != 0:
            print(f"  WARN program dump failed, see {dump_dir}/cinm-opt.log")
        return r

    return _lower


def eval_solution_lowerer(
    *, cinm_opt: pathlib.Path = DEFAULT_CINM_OPT, extra_infer_opts: dict = {}
):
    """A (fn_module, out_file, log_file) -> CompletedProcess callable, for use
    as compile_run.Config.lower -- compiles CINM 2.0's chosen `params` with no
    further search."""

    def _lower(
        fn_module: pathlib.Path,
        out_file: pathlib.Path,
        log_file: pathlib.Path,
        **params,
    ) -> subprocess.CompletedProcess:
        return eval_solution(
            fn_module,
            params,
            out_file=out_file,
            cinm_opt=cinm_opt,
            log_file=log_file,
            extra_infer_opts=extra_infer_opts,
        )

    return _lower
