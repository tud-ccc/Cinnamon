"""doit tasks for the paper evaluation (§8 of the paper).

This pipeline collects the measurements behind the paper's §8: space
dumps, the shared uniform sample, the exhaustive top-k stack, the BO
searches (default + ablated spaces), the transcribed points with their
invariant cross-check, the A2 probe stack, and the assemble/plot layer
that folds whatever exists into results/ and figures. Structure and
conventions follow cinm1comparison/dodo.py: stages connected by files,
fallible-per-config compiles, exclusive hardware benches (one at a time,
in no particular order; the stacks are independent, so any one of them
can be asked for on its own), retry tasks. The shared machinery lives in
cinm_experiments.doit_blocks.

Still to come: the cinm1 (D,T) sweep with coverage accounting, and the
RQ4 multi-op workloads/arms.

Usage:
  doit space             # dump every benchmark's space.json (no simulator)
  doit sample            # draw the shared uniform sample (simulator only)
  doit                   # everything up to bench_sample (hardware!)
  doit retry_failed_compiles && doit
  doit retry_failed_bench && doit bench_sample
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from doit import create_after
from doit.tools import config_changed
from doit.reporter import ProgressBarReporter

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))  # sibling modules (assemble) under doit

from cinm_experiments import cinmopt, compile_run, doit_blocks, pools, ALL_PRIMS  # noqa: E402
from cinm_experiments import space as space_mod  # noqa: E402
from cinm_experiments.paths import DEFAULT_CINM_OPT  # noqa: E402
from cinm_experiments.split_source import list_functions, split_source  # noqa: E402

# ── the paper's constants ────────────────────────────────────────────────────
# These values appear verbatim in the paper (CI arithmetic, A3, RQ3), so they
# are set once here and never inline; each carries its own rationale.
OPTS = dict(
    # 3/n rule of three: 0-of-300 beaten => the point is in the top 1% of the
    # feasible space (95% CI) -- the number the paper quotes.
    n_sample=300,
    # Fixed draw: the sample is THE shared sample (E1(b), RQ3's ground truth,
    # §2's best-vs-median, A2's accepted-but-fails). Changing the seed
    # invalidates every measured row, so it is part of the methodology.
    sample_seed=1815,
    k_top=200,  # B3: measured ground truth for top-k overlap up to k=200
    n_seeds=32,  # B4: matches the cinm1comparison campaign
    iters=6,  # measurement repetitions per hardware run
    # Simulator to use for all trials.
    simulator="cycle-accurate",
    # Simulation wall-clock budget per configuration. The fat tail of the
    # simulator is what this is for: a uniform draw turns up the occasional
    # config whose simulation runs for minutes while the other 299 take
    # milliseconds, and one of those stalls the whole sweep.
    #
    # Capping it does NOT condition the sample, because a timed-out config is
    # kept rather than resampled past: it takes its slot with a non-finite
    # predicted cost and still gets compiled and benchmarked downstream, so
    # the measured distribution the percentile is computed over is still all
    # n_sample draws. Only the predicted column has a hole in it, and
    # sample_stats.json counts the holes so the reporting can state the
    # bound that survives assuming every one of them beats the reference.
    #
    # This is also why sample-max-cost-ms must stay 0 (see _draw_sample):
    # that one rejects and resamples, which is the thing that would bias.
    eval_timeout_ms=10_000,
    # B3's exhaustive predicted sweep only. A timeout is safe here, unlike
    # in B1: it can only mis-price configs slower than 300 ms, which cannot
    # be in the top anyway, and it speeds the sweep up considerably.
    exhaust_timeout_ms=1000,
    # A2's Cartesian draw. Same rule-of-three arithmetic as n_sample: with
    # feasible densities around 1e-10 essentially every draw lands in the
    # rejected region, so 0-lowers-of-300 bounds the false-negative rate of
    # the constraint system at 1% (95%). Its own seed: the draw is over the
    # Cartesian domains, not the feasible set, and must not be conflated
    # with the shared sample.
    n_a2=300,
    a2_seed=1848,
    infer_opts={
        # todo Maybe bump this up for the real numbers
        "bo-batch-size": 4,
        "max-evals": 256,
        "n-init": 64,
        "acquisition": "thompson",
    },
    # The whole-program stack's budget, per (class, device size). Far smaller
    # than infer_opts, because a graph solve spends it once per menu point:
    # llama's 12 device classes over a 64-granularity menu of a 2560-DPU
    # device is ~450 searches where a per-operator run does one. What the
    # profile has to get right is the shape of a class's cost against device
    # size -- the allocator only ever compares points of one class to each
    # other -- not the global best configuration.
    wholeprog_infer_opts={
        "bo-batch-size": 2,
        "max-evals": 256,
        "n-init": 64,
        "acquisition": "thompson",
    },
    # Repeats of each menu point's search, for the noise band on the profile
    # figures. A profile is a row of independent searches, so a difference
    # between two menu points is only real if it clears the spread within
    # them -- and differencing (the marginal-returns panel, the convexity
    # question) amplifies that spread. Diagnostic only: the allocator is
    # handed the first seed's point either way, so this cannot move an
    # allocation. Costs a full profiling sweep per extra seed.
    wholeprog_profile_seeds=16,
)

WORKLOADS = list(ALL_PRIMS)
DATA_DIR = HERE / "data"

DOIT_CONFIG = {
    "default_tasks": ["bench_sample"],
    "verbosity": 2,
    "continue": True,
    "reporter": ProgressBarReporter,
}


# ── paths ────────────────────────────────────────────────────────────────────


def source_mlir(bench: str) -> pathlib.Path:
    return EXPERIMENTS_DIR / f"{bench}.mlir"


def split_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "_split"


def split_module(bench: str, fn_name: str) -> pathlib.Path:
    return split_dir(bench) / f"{fn_name}.mlir"


def space_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "space"


def space_json(bench: str, fn_name: str) -> pathlib.Path:
    return space_dir(bench) / f"infer_{fn_name}" / "space.json"


def sample_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "sample"


def sample_pool_csv(bench: str, fn_name: str) -> pathlib.Path:
    return sample_dir(bench) / f"infer_{fn_name}" / "pool.csv"


def sample_stats_json(bench: str, fn_name: str) -> pathlib.Path:
    """The draw's census: how many candidates were requested, accepted, timed
    out, and rejected, and for which reason. The rejections leave no row in
    pool.csv, so this is the only record that the draw covered the whole
    space -- without it the sample cannot be described as a draw from
    anything, and the percentile has nothing to stand on."""
    return sample_dir(bench) / f"infer_{fn_name}" / "sample_stats.json"


def sample_roots(bench: str) -> doit_blocks.MeasureRoots:
    return doit_blocks.MeasureRoots(
        compile_root=sample_dir(bench) / "compiled",
        run_root=sample_dir(bench) / "run",
    )


# ── one cinm-opt run per function ───────────────────────────────────────────
# Every cinm-opt stage below (space, sample, exhaust, search) is one task per
# function, run over that function's split module rather than the whole
# benchmark. The functions were being processed one after another inside a
# single run anyway, so this costs nothing and buys the granularity: an
# interrupted stage keeps the functions it finished, a function that fails
# does not take its siblings down, and the progress bar names the function
# actually running.
#
# They keep sharing one dump root per (benchmark, stage) so the dumps stay
# where everything downstream already looks for them -- infer_{fn_name}/
# separates those by itself. What does not separate itself is the pass's own
# output and log, hence the per-function names here; losing the log would
# lose which function failed.


def _per_fn_out_mlir(dump: pathlib.Path, fn_name: str) -> pathlib.Path:
    return dump / f"out_{fn_name}.mlir"


def _per_fn_run_files(dump: pathlib.Path, fn_name: str) -> dict:
    """out_file/log_file for a cinm-opt run over one function's split module"""
    return {
        "out_file": _per_fn_out_mlir(dump, fn_name),
        "log_file": dump / f"cinm-opt_{fn_name}.log",
    }


# ── split ────────────────────────────────────────────────────────────────────


def _split_one(src: pathlib.Path, out_dir: pathlib.Path) -> bool:
    split_source(src, out_dir)
    return True


def task_split():
    """Split each benchmark's source into one module per function."""
    for bench in WORKLOADS:
        src = source_mlir(bench)
        fns = list_functions(src)
        yield {
            "name": bench,
            "file_dep": [str(src)],
            "targets": [str(split_module(bench, fn)) for fn in fns],
            "actions": [(_split_one, [src, split_dir(bench)])],
        }


# ── B0: space dumps ──────────────────────────────────────────────────────────


def _dump_space(bench: str, fn_name: str) -> bool:
    cinmopt.dump_space(
        split_module(bench, fn_name),
        space_dir(bench),
        **_per_fn_run_files(space_dir(bench), fn_name),
    )
    return True


def task_space():
    """B0: dump every function's space.json -- sizes for tab:sufficiency,
    per-param docs + permutation tables for the manual ATiM transcription
    workflow. No simulator runs; safe anywhere."""
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            yield {
                "name": f"{bench}:{fn_name}",
                "file_dep": [str(split_module(bench, fn_name))],
                "targets": [str(space_json(bench, fn_name))],
                "actions": [(_dump_space, [bench, fn_name])],
            }


# ── B1: the shared uniform sample ────────────────────────────────────────────


def _draw_sample(bench: str, fn_name: str) -> bool:
    cinmopt.random_sample(
        split_module(bench, fn_name),
        sample_dir(bench),
        workers=64,
        n_samples=OPTS["n_sample"],
        seed=OPTS["sample_seed"],
        **_per_fn_run_files(sample_dir(bench), fn_name),
        infer_opts={
            "simulator": OPTS["simulator"],
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            # Must stay 0. This one rejects a candidate and draws another in
            # its place, which conditions the sample on the accepted region
            # and voids the percentile arithmetic. eval-timeout-ms bounds the
            # wall clock without doing that (see OPTS).
            "sample-max-cost-ms": 0,
            # Independent uniform draws, not LHS: this sample is read as a
            # picture of the space -- a percentile, a rank correlation, a
            # confidence bound -- and LHS stratifies the picks, so they are
            # not the independent trials that arithmetic assumes.
            "sampling-mode": "uniform",
        },
    )
    return True


def task_sample():
    """B1: draw the shared uniform sample -- ONE set of n_sample configs per
    function, drawn uniformly at random from the enumerated feasible set
    with a fixed seed, predicted costs recorded. Serves E1(b), RQ3's
    fidelity ground truth, §2's best-vs-median, and A2's accepted-but-fails
    check all at once; nothing downstream may redraw it.

    A config whose simulation exceeds eval_timeout_ms keeps its slot with a
    non-finite predicted cost, so all n_sample rows still reach B2 and the
    measured distribution stays whole; sample_stats.json counts them.

    Splitting the draw per function does not change it: the pass seeds a
    fresh RNG per function from sample_seed, so a function's rows do not
    depend on which functions ran before it."""
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            yield {
                "name": f"{bench}:{fn_name}",
                "file_dep": [str(split_module(bench, fn_name))],
                "targets": [
                    str(sample_pool_csv(bench, fn_name)),
                    str(sample_stats_json(bench, fn_name)),
                ],
                "actions": [(_draw_sample, [bench, fn_name])],
            }


# ── B2 on the sample: compile + bench every sampled config ──────────────────


def _sample_configs(bench: str) -> list[compile_run.Config]:
    """One Config per sampled row, labelled by its row position ("s000",
    "s001", ...) -- positions are stable because the sample itself is (fixed
    seed, never redrawn)."""
    configs = []
    for fn_name in list_functions(source_mlir(bench)):
        pool_csv = sample_pool_csv(bench, fn_name)
        for i, params in enumerate(pools.sample_rows(pool_csv)):
            configs.append(
                compile_run.Config(
                    system="sample",
                    fn_name=fn_name,
                    label=f"s{i:03d}",
                    params=params,
                    fn_module=split_module(bench, fn_name),
                    prim=bench.removeprefix("prim_"),
                    lower=cinmopt.eval_solution_lowerer(),
                )
            )
    return configs


# One entry of a measure stack: which benchmark it belongs to, the config
# to compile+bench, the files whose change invalidates it, and its roots.
StackEntry = tuple[
    str, "compile_run.Config", list[pathlib.Path], "doit_blocks.MeasureRoots"
]


def _measure_stack(
    stack: str,
    entries: list[StackEntry],
    *,
    qualify_by_system: bool = False,
):
    """Yield compile_{stack} / bench_{stack} tasks for a list of configs.

    `qualify_by_system` puts the config's system in the task name. A stack
    drawing on several sources needs it -- the transcriptions of ATiM's
    published and reproduced schedules are different configs that share a
    benchmark, a function and a candidate label -- and a stack with one
    source must not have it, or its task names change under it.

    Compiles are parallel and fallible per config. Benches are `exclusive`,
    so hardware only ever runs one config at a time, in no particular order:
    stacks are independent of each other, so any one of them can be asked
    for on its own without dragging the others onto the hardware first.
    """

    def task_name(bench, config) -> str:
        parts = [bench, config.fn_name, config.label]
        if qualify_by_system:
            parts.insert(1, config.system)
        return ":".join(parts)

    for bench, config, deps, roots in entries:
        name = task_name(bench, config)
        marker = roots.compile_marker_of(config)
        yield {
            "basename": f"compile_{stack}",
            "name": name,
            "file_dep": [str(d) for d in deps] + [str(config.fn_module)],
            "targets": [str(marker)],
            "actions": [(doit_blocks.compile_one, [config, roots, marker])],
        }
        bench_marker = roots.bench_marker_of(config)
        yield {
            "basename": f"bench_{stack}",
            "name": name,
            # The stack inputs too: a redrawn pool / rewritten point must
            # invalidate the measurements, not silently re-attribute them.
            "file_dep": [str(marker)] + [str(d) for d in deps],
            # Real hardware: never beside anything else.
            "exclusive": True,
            "targets": [str(bench_marker)],
            "actions": [
                (
                    doit_blocks.bench_one_config,
                    [config, roots],
                    dict(iters=OPTS["iters"], bench_marker=bench_marker),
                )
            ],
        }


@create_after(executed="sample", creates=["compile_sample", "bench_sample"])
def task_compile_sample():
    """Compile + bench every sampled config. An eval-solution compile
    failure here is accepted-but-fails DATA (the paper says it must be 0),
    not just noise -- keep the logs."""
    entries: list[StackEntry] = []
    for bench in WORKLOADS:
        for config in _sample_configs(bench):
            entries.append(
                (
                    bench,
                    config,
                    [sample_pool_csv(bench, config.fn_name)],
                    sample_roots(bench),
                )
            )
    yield from _measure_stack("sample", entries)


# ── A2: rejected-region probing (CPU-only, no hardware) ─────────────────────


def a2_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "a2"


def a2_probe_csv(bench: str, fn_name: str) -> pathlib.Path:
    return a2_dir(bench) / fn_name / "probe.csv"


def a2_done_marker(bench: str, fn_name: str) -> pathlib.Path:
    return a2_dir(bench) / fn_name / "probe.done"


def _probe_a2(bench: str, fn_name: str) -> bool:
    """Draw n_a2 assignments uniformly from the Cartesian domains of this
    function's space and force-lower each; probe.csv records one verdict
    per draw. Membership and lowering verdict both come from the single
    forced cinm-opt run (see cinmopt.probe_solution), so no materialised
    feasible pool is needed. Resumable: rows already present are kept, the
    draw is deterministic (fixed seed), and the .done marker only appears
    once every draw has a row."""
    space = space_mod.load(space_json(bench, fn_name))
    draws = space.sample_cartesian(OPTS["n_a2"], seed=OPTS["a2_seed"])

    csv_path = a2_probe_csv(bench, fn_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dims = space.dim_names
    header = ",".join(["label", *dims, "verdict"])

    done: set[str] = set()
    if csv_path.exists():
        lines = csv_path.read_text().splitlines()
        if lines and lines[0] != header:
            # The space (or the draw) changed under the file: stale rows
            # would silently mis-attribute verdicts, so start over.
            lines = []
            csv_path.unlink()
        done = {line.split(",", 1)[0] for line in lines[1:]}

    logs_dir = csv_path.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(csv_path, "a") as out:
        if not done:
            out.write(header + "\n")
        for i, params in enumerate(draws):
            label = f"a{i:03d}"
            if label in done:
                continue
            verdict = cinmopt.probe_solution(
                split_module(bench, fn_name),
                params,
                log_file=logs_dir / f"{label}.log",
            )
            out.write(
                ",".join([label, *(str(params[d]) for d in dims), verdict]) + "\n"
            )
            out.flush()
    a2_done_marker(bench, fn_name).touch()
    return True


def task_a2():
    """A2 probe: per function, force-lower a Cartesian sample from outside
    (mostly) the feasible set. CPU-only and parallel across benchmarks
    (`doit -n 7 a2`); a draw that happens to land feasible contributes to
    the accepted-but-fails counter instead of the rejected-region one."""
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            yield {
                "name": f"{bench}:{fn_name}",
                "file_dep": [
                    str(space_json(bench, fn_name)),
                    str(split_module(bench, fn_name)),
                ],
                "targets": [str(a2_done_marker(bench, fn_name))],
                "actions": [(_probe_a2, [bench, fn_name])],
            }


RESULTS_DIR = HERE / "results"
OFFLINE_DIR = HERE / "offline"


# ── stack path conventions ──────────────────────────────────────────────────
# The stacks below B1 are not all wired yet; their paths are fixed here so
# the assemble layer can already glob them and report MISSING, and so the
# future task families land in agreed places.


def topk_roots(bench: str) -> doit_blocks.MeasureRoots:
    return doit_blocks.MeasureRoots(
        compile_root=DATA_DIR / bench / "topk" / "compiled",
        run_root=DATA_DIR / bench / "topk" / "run",
    )


def search_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "search"


def search_roots(bench: str) -> doit_blocks.MeasureRoots:
    return doit_blocks.MeasureRoots(
        compile_root=search_dir(bench) / "compiled",
        run_root=search_dir(bench) / "run",
    )


def ablate_roots(bench: str, space: str) -> doit_blocks.MeasureRoots:
    root = DATA_DIR / bench / f"search_ablate_{space}"
    return doit_blocks.MeasureRoots(
        compile_root=root / "compiled", run_root=root / "run"
    )


ABLATE_SPACES = ["no_mram", "no_scatter", "neither"]


def points_roots(bench: str, source: str) -> doit_blocks.MeasureRoots:
    root = DATA_DIR / bench / f"points_{source}"
    return doit_blocks.MeasureRoots(
        compile_root=root / "compiled", run_root=root / "run"
    )


RQ4_PROGRAMS: list[str] = []  # filled when the multi-op workloads land


def rq4_roots(prog: str, arm: str) -> doit_blocks.MeasureRoots:
    root = DATA_DIR / prog / f"rq4_{arm}"
    return doit_blocks.MeasureRoots(
        compile_root=root / "compiled", run_root=root / "run"
    )


# ── whole-program graph allocation ──────────────────────────────────────────
# The RQ4 workloads as the graph solver sees them: a whole program, unrolled
# so every layer is its own set of compute blocks, run through the two-level
# solve. No hardware and no measurement -- this stack produces the solver's
# own view (per-class cost profiles, the device sets it carved out) and is
# what plot_profiles.py draws. Measuring these programs is the RQ4 arms
# above, which is a separate (and still unwired) stack.
#
# The front end lives in each program's own makefile, so the pipeline is
# written down once and stays runnable by hand; the tasks below only pass it
# the campaign's cinm-opt and an output directory under data/.

WHOLEPROG_DIR = HERE / "wholeprogram"
# program -> the directory holding its source and makefile. The program name
# is the makefile's NAME, i.e. its source is <dir>/<prog>.mlir.
WHOLEPROG_PROGRAMS = {"llama2_110M": WHOLEPROG_DIR / "llama2"}


def wholeprog_dir(prog: str) -> pathlib.Path:
    return DATA_DIR / "wholeprog" / prog


def wholeprog_front_mlir(prog: str) -> pathlib.Path:
    """The front end's last stage: compute blocks formed and the host code
    between them wrapped, so the function is one connected dataflow graph."""
    return wholeprog_dir(prog) / f"{prog}_with_compute.mlir"


def wholeprog_alloc_dir(prog: str) -> pathlib.Path:
    return wholeprog_dir(prog) / "alloc"


def _wholeprog_front(prog: str) -> bool:
    out_dir = wholeprog_dir(prog)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "make",
            "-C",
            str(WHOLEPROG_PROGRAMS[prog]),
            f"NAME={prog}",
            f"OUT={out_dir.resolve()}",
            f"CINM_OPT={DEFAULT_CINM_OPT}",
            str(wholeprog_front_mlir(prog).resolve()),
        ],
        check=True,
    )
    return True


def task_wholeprog_front():
    """The whole-program front end (unroll, linalg, compute blocks) for each
    RQ4-style program, via the program's own makefile. The unroll is the
    expensive stage and the reason this is its own task: it turns the layer
    and head loops into straight-line code, which is what gives the graph
    solver a class per distinct kernel and a member per instance."""
    for prog, src_dir in WHOLEPROG_PROGRAMS.items():
        yield {
            "name": prog,
            "file_dep": [str(src_dir / f"{prog}.mlir"), str(src_dir / "Makefile")],
            "targets": [str(wholeprog_front_mlir(prog))],
            "actions": [(_wholeprog_front, [prog])],
        }


def _wholeprog_alloc(prog: str) -> bool:
    cinmopt.graph_allocation(
        wholeprog_front_mlir(prog),
        wholeprog_alloc_dir(prog),
        workers=64,
        infer_opts={
            "simulator": OPTS["simulator"],
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            **OPTS["wholeprog_infer_opts"],
            "profile-seeds": OPTS["wholeprog_profile_seeds"],
            # Stamping commits every member's configuration as attributes on
            # one module (--upmem-lower-stamped lowers it afterwards), which
            # is the only commit path that scales to a whole program.
            "stamp-configs": True,
            # Without it the f32-reduction-only blocks (rmsnorm's sum of
            # squares, softmax's max and sum) have no feasible configuration
            # at all and drop to the host, which would take the classes that
            # matter most out of the picture.
            "allow-float-reassociation": True,
        },
    )
    return True


def task_wholeprog_alloc():
    """The two-level graph solve over each whole program: a cost profile per
    class over the device-size menu, then the device divided between the
    classes. Dumps profiles.csv / allocation.csv / groups.csv per graph;
    simulator only, no hardware.

    out.mlir (the stamped module) doubles as the completion sentinel: the
    dumps are per graph and named by the pass, so there is no single dump
    path known ahead of time to hang the target on."""
    for prog in WHOLEPROG_PROGRAMS:
        yield {
            "name": prog,
            "file_dep": [str(wholeprog_front_mlir(prog))],
            "targets": [str(wholeprog_alloc_dir(prog) / "out.mlir")],
            "uptodate": [config_changed(OPTS["wholeprog_infer_opts"])],
            "actions": [(_wholeprog_alloc, [prog])],
        }


def task_plot_alloc():
    """plots/wholeprog/<prog>/profiles.pdf: each program's per-class cost
    profiles against device size, annotated with what the allocator did with
    them. Reads the dumps directly (they are solver artifacts, not assembled
    measurements) and skips with a note when a program has not been solved
    yet, so it is safe to ask for at any point."""
    for prog in WHOLEPROG_PROGRAMS:
        yield {
            "name": prog,
            "actions": [
                f"python {HERE / 'plot_profiles.py'} {wholeprog_alloc_dir(prog)}"
                f" --out {HERE / 'plots' / 'wholeprog' / prog}"
            ],
            "uptodate": [False],
        }


# ── B6: assemble results/*.csv from whatever exists ─────────────────────────

import assemble  # noqa: E402  (sibling module; HERE is on sys.path via doit)


def _pair(
    roots: doit_blocks.MeasureRoots, system: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """The (compile, run) roots the assemblers iterate. MeasureRoots nests a
    <system> directory under each root (compile_root/<system>/<fn>/<label>),
    while the assemble layer walks <fn>/<label> -- so the system level is
    consumed here, not guessed there."""
    return (roots.compile_root / system, roots.run_root / system)


def _assemble_e1() -> bool:
    return assemble.assemble_e1(
        {
            bench: {
                "sample": _pair(sample_roots(bench), "sample"),
                "topk": _pair(topk_roots(bench), "topk"),
                "search": _pair(search_roots(bench), "search"),
                **{
                    POINT_SOURCES[source]: _pair(
                        points_roots(bench, source), POINT_SOURCES[source]
                    )
                    for source in ("atim_published", "atim_reproduced")
                },
            }
            for bench in WORKLOADS
        },
        OFFLINE_DIR / "atim.csv",
        RESULTS_DIR / "e1.csv",
    )


def _assemble_rq1() -> bool:
    return assemble.assemble_rq1(
        {
            bench: {
                "ours": _pair(search_roots(bench), "search"),
                "cinm1": (
                    DATA_DIR / bench / "cinm1" / "compiled" / "cinm1",
                    DATA_DIR / bench / "cinm1" / "run" / "cinm1",
                ),
                "cinm1_rule": _pair(points_roots(bench, "cinm1rule"), "cinm1_rule"),
            }
            for bench in WORKLOADS
        },
        {system: OFFLINE_DIR / f"{system}.csv" for system in ("prim", "atim", "cpu")},
        RESULTS_DIR / "rq1.csv",
    )


def _assemble_rq2() -> bool:
    return assemble.assemble_rq2(
        # the dump root, not search_dir: the timings live under
        # {dump}/infer_{fn}/seed_{k}/, which is what assemble_rq2 globs
        {bench: _search_dump_dir(bench, "default") for bench in WORKLOADS},
        {
            (bench, fn): space_json(bench, fn)
            for bench in WORKLOADS
            for fn in list_functions(source_mlir(bench))
        },
        RESULTS_DIR / "rq2.csv",
        OFFLINE_DIR / "atim.csv",
    )


def _assemble_rq3() -> bool:
    return assemble.assemble_rq3(
        {bench: _pair(sample_roots(bench), "sample") for bench in WORKLOADS},
        RESULTS_DIR / "rq3.csv",
    )


def _assemble_sample_census() -> bool:
    keys = [
        (bench, fn) for bench in WORKLOADS for fn in list_functions(source_mlir(bench))
    ]
    return assemble.assemble_sample_census(
        {k: sample_stats_json(*k) for k in keys},
        {k: sample_pool_csv(*k) for k in keys},
        RESULTS_DIR / "sample_census.csv",
    )


def _assemble_a1() -> bool:
    return assemble.assemble_a1(
        {
            (bench, space): _pair(
                search_roots(bench)
                if space == "default"
                else ablate_roots(bench, space),
                "search",
            )
            for bench in WORKLOADS
            for space in ["default", *ABLATE_SPACES]
        },
        RESULTS_DIR / "a1.csv",
    )


def _assemble_rq4() -> bool:
    return assemble.assemble_rq4(
        {
            (prog, arm): _pair(rq4_roots(prog, arm), arm)
            for prog in RQ4_PROGRAMS
            for arm in ("peroper", "wholeprog")
        },
        RESULTS_DIR / "rq4.csv",
    )


def task_assemble():
    """B6: one sub-task per results table, each globbing whatever B1-B5
    produced and printing MISSING notes instead of failing."""
    for name, action in [
        ("e1", _assemble_e1),
        ("rq1", _assemble_rq1),
        ("rq2", _assemble_rq2),
        ("rq3", _assemble_rq3),
        ("a1", _assemble_a1),
        ("rq4", _assemble_rq4),
        ("sample_census", _assemble_sample_census),
    ]:
        yield {
            "name": name,
            "actions": [action],
            "uptodate": [False],  # missing-tolerant: always re-derive
        }


PLOT_SCRIPTS = [
    "table_sufficiency.py",
    "plot_sufficiency.py",
    "plot_quality.py",
    "table_quality.py",
    "table_walltime.py",
    "plot_fidelity.py",
    "plot_wholeprogram.py",
    "table_capability.py",
]


def task_plots():
    """Every plot/table derivable from the current
    results/ -- each script skips (exit 0) when its input CSV is not
    assembled yet, so `doit plots` is safe at any stage of the campaign."""
    for script in PLOT_SCRIPTS:
        yield {
            "name": pathlib.Path(script).stem,
            "actions": [f"python {HERE / script}"],
            "task_dep": ["assemble", "assemble_a2"],
            "uptodate": [False],
        }


def _assemble_a2() -> bool:
    """results/a2.csv: per-function verdict counts. Missing-tolerant
    : functions not yet probed are simply absent, and partial
    probe.csvs contribute the rows they have."""
    import csv

    rows = []
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            probe = a2_probe_csv(bench, fn_name)
            if not probe.exists():
                continue
            counts = {
                "accepted": 0,
                "accepted_fails": 0,
                "rejected_lowers": 0,
                "rejected_fails": 0,
            }
            with open(probe) as f:
                for row in csv.DictReader(f):
                    if row["verdict"] in counts:
                        counts[row["verdict"]] += 1
            rows.append(
                {
                    "bench": bench,
                    "fn": fn_name,
                    "n_probed": sum(counts.values()),
                    **{f"n_{k}": v for k, v in counts.items()},
                }
            )
    if not rows:
        print("assemble_a2: no probe.csv present yet, nothing to assemble")
        return True
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "a2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return True


def task_assemble_a2():
    """Assemble results/a2.csv from whatever probes exist."""
    return {
        "actions": [_assemble_a2],
        "uptodate": [False],  # missing-tolerant: always re-derive
    }


def task_numbers():
    """tables/numbers.tex: the paper's inline numbers as \\newcommand defs,
    from whatever results/*.csv exist (the script is
    paper_numbers.py -- a numbers.py here would shadow the stdlib module)."""
    return {
        "actions": [f"python {HERE / 'paper_numbers.py'}"],
        "task_dep": ["assemble_a2"],
        "uptodate": [False],
    }


# ── retries ──────────────────────────────────────────────────────────────────


def _retry_failed_compiles() -> bool:
    for bench in WORKLOADS:
        doit_blocks.clear_failed_compiles(_sample_configs(bench), sample_roots(bench))
    return True


def task_retry_failed_compiles():
    """Not part of the default pipeline; run explicitly after fixing the
    cause, then rerun `doit`."""
    return {"actions": [_retry_failed_compiles], "uptodate": [False]}


def _retry_failed_bench() -> bool:
    for bench in WORKLOADS:
        doit_blocks.clear_failed_bench(_sample_configs(bench), sample_roots(bench))
    return True


def task_retry_failed_bench():
    """Not part of the default pipeline; run explicitly after fixing the
    cause, then rerun `doit bench_sample`."""
    return {"actions": [_retry_failed_bench], "uptodate": [False]}


# ── re-pricing what is already compiled ─────────────────────────────────────


def _compiled_config_dirs() -> list[tuple[str, pathlib.Path]]:
    """(prim, config_dir) for every compiled config under data/, whichever
    stack it belongs to. Every stack lays its compiles out the same way --
    <bench>/<stack>/compiled/<system>/<fn_name>/<label> -- so one glob finds
    them all, including the stacks whose task family has not landed yet.

    The filesystem is the enumeration here, not the config lists the compile
    tasks are built from: this re-prices what was compiled, which is not
    always what the current dodo would choose to compile."""
    dirs = []
    for lowered in sorted(DATA_DIR.glob("*/*/compiled/*/*/*/lowered.mlir")):
        bench = lowered.relative_to(DATA_DIR).parts[0]
        dirs.append((bench.removeprefix("prim_"), lowered.parent))
    return dirs


def _recost_one(config_dir: pathlib.Path, prim: str) -> bool:
    err = compile_run.recompute_cost(config_dir, prim=prim)
    if err:
        print(f"  FAIL cost: {config_dir}: {err}")
        return False
    return True


def task_recost():
    """Re-run the cost model over every already-compiled config and rewrite
    its ir/cost.csv. Lowers nothing, compiles no DPU kernel, links no binary
    and touches no hardware -- just the --upmem-annotate-costs pass over the
    lowered.mlir already on disk.

    This is the task to run when the cost model itself changes (a fix, a
    refit of the latency tables) and the predictions have to catch up with
    it. The measured side is untouched and stays valid, because the binaries
    and the schedules they were built from have not changed -- only what we
    predict about them. Follow it with `doit assemble` to fold the new
    predictions into results/rq3.csv.

    Keyed on the cinm-opt binary, so it is up to date until the compiler is
    rebuilt and then re-prices everything exactly once. Not in
    default_tasks: a rebuild should not silently re-price the sample in the
    middle of someone else's run.

    Per-config and fallible: one config that fails to price is reported and
    leaves its stale cost.csv behind, and (with `continue`) the rest still
    run."""
    for prim, config_dir in _compiled_config_dirs():
        yield {
            "name": str(config_dir.relative_to(DATA_DIR)),
            "file_dep": [str(config_dir / "lowered.mlir"), str(DEFAULT_CINM_OPT)],
            "targets": [str(config_dir / "ir" / "cost.csv")],
            "actions": [(_recost_one, [config_dir, prim])],
        }


# ── B3: best-of-space by the cost model ─────────────────────────────────────


def exhaust_dir(bench: str) -> pathlib.Path:
    return DATA_DIR / bench / "topk" / "exhaust"


def exhaust_pool_csv(bench: str, fn_name: str) -> pathlib.Path:
    return exhaust_dir(bench) / f"infer_{fn_name}" / "pool.csv"


def _exhaust_one(bench: str, fn_name: str) -> bool:
    cinmopt.exhaustive_search(
        split_module(bench, fn_name),
        exhaust_dir(bench),
        **_per_fn_run_files(exhaust_dir(bench), fn_name),
        infer_opts={
            # Same model pairing as B1/B4 -- predicted costs must be
            # comparable across stacks -- but with the shorter sweep
            # timeout: it can only mis-price configs that cannot compete
            # for the top anyway.
            "simulator": OPTS["simulator"],
            "eval-timeout-ms": OPTS["exhaust_timeout_ms"],
        },
    )
    return True


def task_exhaust_pred():
    """B3, predicted half: exhaustively price every feasible configuration
    (no hardware, CPU-heavy). The full pool is also the feasible-set oracle
    for whatever wants membership with costs attached."""
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            yield {
                "name": f"{bench}:{fn_name}",
                "file_dep": [str(split_module(bench, fn_name))],
                "targets": [str(exhaust_pool_csv(bench, fn_name))],
                "actions": [(_exhaust_one, [bench, fn_name])],
            }


def _topk_configs(bench: str) -> list[compile_run.Config]:
    configs = []
    for fn_name in list_functions(source_mlir(bench)):
        pool_csv = exhaust_pool_csv(bench, fn_name)
        if not pool_csv.exists():
            continue
        for i, params in enumerate(pools.top_k(pool_csv, OPTS["k_top"])):
            configs.append(
                compile_run.Config(
                    system="topk",
                    fn_name=fn_name,
                    label=f"k{i:03d}",
                    params=params,
                    fn_module=split_module(bench, fn_name),
                    prim=bench.removeprefix("prim_"),
                    lower=cinmopt.eval_solution_lowerer(),
                )
            )
    return configs


@create_after(executed="exhaust_pred", creates=["compile_topk", "bench_topk"])
def task_compile_topk():
    """B3, measured half: compile + bench the k_top best-predicted configs
    per function. Their measured costs ground the top-k-overlap metric
    exactly (instead of only within the sample) and feed best-of-
    {sample U topk}. Labels are rank order ("k000" = predicted best)."""
    entries: list[StackEntry] = []
    for bench in WORKLOADS:
        for config in _topk_configs(bench):
            entries.append(
                (
                    bench,
                    config,
                    [exhaust_pool_csv(bench, config.fn_name)],
                    topk_roots(bench),
                )
            )
    yield from _measure_stack("topk", entries)


# ── B4: the BO search stack, default + ablated spaces ───────────────────────

# The capability each restricted space turns off, as pass options. The
# space itself is what changes (use-mram-tiling constrains MRAM tile ==
# WRAM tile; the scatter toggle disables both specialisation sites in the
# trial lowering); the search machinery is identical.
ABLATE_INFER_OPTS = {
    "no_mram": {"use-mram-tiling": False},
    "no_scatter": {"enable-scatter-specialisation": False},
    "neither": {"use-mram-tiling": False, "enable-scatter-specialisation": False},
}


def _search_dump_dir(bench: str, space: str) -> pathlib.Path:
    return (
        search_dir(bench) / "dump"
        if space == "default"
        else DATA_DIR / bench / f"search_ablate_{space}" / "dump"
    )


def _search_out_mlir(bench: str, space: str, fn_name: str) -> pathlib.Path:
    """The search's compiled output, which doubles as the task's sentinel that
    this function's search ran to completion -- unlike the other stages, a
    search has no per-function dump that only exists once it is finished."""
    return _per_fn_out_mlir(_search_dump_dir(bench, space), fn_name)


def _run_search(bench: str, fn_name: str, space: str) -> bool:
    dump = _search_dump_dir(bench, space)
    cinmopt.bo_multiseed(
        split_module(bench, fn_name),
        dump,
        n_seeds=OPTS["n_seeds"],
        workers=64,
        **_per_fn_run_files(dump, fn_name),
        infer_opts={
            "simulator": OPTS["simulator"],
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            **OPTS["infer_opts"],
            **ABLATE_INFER_OPTS.get(space, {}),
        },
    )
    return True


def task_search():
    """B4: the multi-seed BO search over the default space (no hardware;
    simulator only). Each seed's pick is compiled+benched downstream; the
    per-seed timings.csv is the search-walltime raw data.

    One task per function, searching that function's split module: a
    benchmark's functions are searched one after the other anyway, and
    separating them means an interrupted run keeps the functions it
    finished, and a failing one does not take the rest of the benchmark
    down with it."""
    for bench in WORKLOADS:
        for fn_name in list_functions(source_mlir(bench)):
            yield {
                "name": f"{bench}:{fn_name}",
                "file_dep": [str(split_module(bench, fn_name))],
                "targets": [str(_search_out_mlir(bench, "default", fn_name))],
                "actions": [(_run_search, [bench, fn_name, "default"])],
            }


def task_search_ablate():
    """B4 over the three restricted spaces -- the capability ablation. An
    ablated search that finds nothing feasible is a result (the capability
    buys feasibility), which the assembly records as such."""
    for bench in WORKLOADS:
        for space in ABLATE_SPACES:
            for fn_name in list_functions(source_mlir(bench)):
                yield {
                    "name": f"{bench}:{space}:{fn_name}",
                    "file_dep": [str(split_module(bench, fn_name))],
                    "targets": [str(_search_out_mlir(bench, space, fn_name))],
                    "actions": [(_run_search, [bench, fn_name, space])],
                }


# ── B7: search-strategy campaign (docs/SearchStrategyPlan.md) ────────────────

# Arm → infer-opts overrides on top of OPTS["infer_opts"]. The `bananas`
# control arm is not listed: it is task_search's own dump, which the
# assembly reads as arm "bananas" (the strategy refactor kept the default
# search bit-identical, so re-running it would only burn compute). Keep
# assemble_campaign.py's ARM_DUMPS in sync with this table.
CAMPAIGN_ARMS = {
    "bananas_rand": {"n-random-candidates": 256},
    "random": {"search-strategy": "random"},
    "descent": {"search-strategy": "descent"},
    "ga": {"search-strategy": "ga"},
}


def campaign_dump_dir(bench: str, arm: str) -> pathlib.Path:
    return (
        _search_dump_dir(bench, "default")
        if arm == "bananas"
        else DATA_DIR / bench / f"campaign_{arm}" / "dump"
    )


def _run_campaign_arm(bench: str, fn_name: str, arm: str) -> bool:
    dump = campaign_dump_dir(bench, arm)
    cinmopt.bo_multiseed(
        split_module(bench, fn_name),
        dump,
        n_seeds=OPTS["n_seeds"],
        workers=64,
        **_per_fn_run_files(dump, fn_name),
        infer_opts={
            "simulator": OPTS["simulator"],
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            **OPTS["infer_opts"],
            **CAMPAIGN_ARMS[arm],
        },
    )
    return True


def task_campaign():
    """B7: the strategy-comparison arms (docs/SearchStrategyPlan.md), same
    seeds and budget as task_search so the shared init sample makes the
    comparison paired. Simulator only; the regret analysis never leaves sim
    space, so nothing here is compiled or benched downstream."""
    for bench in WORKLOADS:
        for arm in CAMPAIGN_ARMS:
            for fn_name in list_functions(source_mlir(bench)):
                dump = campaign_dump_dir(bench, arm)
                yield {
                    "name": f"{bench}:{arm}:{fn_name}",
                    "file_dep": [str(split_module(bench, fn_name))],
                    "targets": [str(_per_fn_out_mlir(dump, fn_name))],
                    "actions": [(_run_campaign_arm, [bench, fn_name, arm])],
                }


def task_assemble_campaign():
    """results/campaign.csv (+ campaign_ref.csv): the anytime curves of
    whatever arms have run, regret-referenced against the pooled best.
    Missing-tolerant like every assemble step."""
    return {
        "actions": [f"python {HERE / 'assemble_campaign.py'}"],
        "uptodate": [False],
    }


def task_plot_campaign():
    """plots/campaign_*.pdf: anytime regret (evals and CPU-seconds), final
    regret, reliability, best-of-k -- see plot_search_campaign.py."""
    return {
        "actions": [f"python {HERE / 'plot_search_campaign.py'}"],
        "task_dep": ["assemble_campaign"],
        "uptodate": [False],
    }


def _search_pick_configs(bench: str, space: str) -> list[compile_run.Config]:
    """One config per (fn, seed): the seed's best-by-predicted-cost pick.
    Every seed is kept -- the spread of the picks over seeds is itself a
    reported number, so deduplicating identical picks would erase it."""
    configs = []
    dump = _search_dump_dir(bench, space)
    if not dump.exists():
        return configs
    for fn_name, seed, params in pools.best_per_seed(dump):
        configs.append(
            compile_run.Config(
                system="search",
                fn_name=fn_name,
                label=f"seed{int(seed):02d}",
                params=params,
                fn_module=split_module(bench, fn_name),
                prim=bench.removeprefix("prim_"),
                lower=cinmopt.eval_solution_lowerer(),
            )
        )
    return configs


@create_after(executed="search", creates=["compile_search", "bench_search"])
def task_compile_search():
    """Compile + bench every seed's pick from the default-space search --
    the flow's own answer, and the seed-spread raw data."""
    entries: list[StackEntry] = []
    for bench in WORKLOADS:
        for config in _search_pick_configs(bench, "default"):
            entries.append(
                (
                    bench,
                    config,
                    [_search_out_mlir(bench, "default", config.fn_name)],
                    search_roots(bench),
                )
            )
    yield from _measure_stack("search", entries)


@create_after(
    executed="search_ablate", creates=["compile_search_ablate", "bench_search_ablate"]
)
def task_compile_search_ablate():
    """Compile + bench the ablated searches' picks, one stack for all
    three restricted spaces."""
    entries: list[StackEntry] = []
    for bench in WORKLOADS:
        for space in ABLATE_SPACES:
            for config in _search_pick_configs(bench, space):
                entries.append(
                    (
                        bench,
                        config,
                        [_search_out_mlir(bench, space, config.fn_name)],
                        ablate_roots(bench, space),
                    )
                )
    yield from _measure_stack("search_ablate", entries)


# ── B5: manually-authored points, evaluated in our system ───────────────────

POINTS_DIR = HERE / "points"
# ATiM ships tuned schedules with its artifact and we also reproduced them by
# tuning on this machine. They are transcribed from different traces into
# different points, so they are separate sources: E1 decomposes the gap to a
# specific ATiM configuration, and which one has to stay visible.
POINT_SOURCES = {
    "atim_published": "atim_published_transcribed",
    "atim_reproduced": "atim_reproduced_transcribed",
    "cinm1rule": "cinm1_rule",
}


def _load_points(source: str, bench: str) -> list[dict]:
    """The checked-in transcription of `bench` from points/{source}/, or []
    when it has not been authored yet. See points/README.md for the file
    format; several candidates per point are the answer to transcription
    ambiguity (measure every reading)."""
    import json

    path = POINTS_DIR / source / f"{bench}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _point_configs(source: str, bench: str) -> list[compile_run.Config]:
    configs = []
    for point in _load_points(source, bench):
        for candidate in point["candidates"]:
            configs.append(
                compile_run.Config(
                    system=POINT_SOURCES[source],
                    fn_name=point["fn_name"],
                    label=candidate["label"],
                    params=candidate["params"],
                    fn_module=split_module(bench, point["fn_name"]),
                    prim=bench.removeprefix("prim_"),
                    lower=cinmopt.eval_solution_lowerer(),
                )
            )
    return configs


def task_compile_points():
    """B5: compile + bench every checked-in transcription candidate
    (ATiM's tuned schedule, CINM 1.0's rule decision) through the
    eval-solution path. Files that don't exist yet contribute nothing."""
    entries: list[StackEntry] = []
    for source in POINT_SOURCES:
        for bench in WORKLOADS:
            for config in _point_configs(source, bench):
                entries.append(
                    (
                        bench,
                        config,
                        [POINTS_DIR / source / f"{bench}.json"],
                        points_roots(bench, source),
                    )
                )
    yield from _measure_stack("points", entries, qualify_by_system=True)


def _invariants_report(source: str, bench: str) -> bool:
    import invariants

    any_report = False
    for point in _load_points(source, bench):
        for candidate in point["candidates"]:
            roots = points_roots(bench, source)
            lowered = (
                roots.config_dir(
                    POINT_SOURCES[source], point["fn_name"], candidate["label"]
                )
                / "lowered.mlir"
            )
            if not lowered.exists():
                print(f"  not compiled yet: {point['fn_name']} {candidate['label']}")
                continue
            report = invariants.report(lowered, candidate.get("expected", {}))
            out = lowered.parent / "invariants.txt"
            out.write_text(report)
            print(f"── {bench}:{point['fn_name']}:{candidate['label']}")
            print(report)
            any_report = True
    if not any_report:
        print(f"no compiled candidates for {source}/{bench}")
    return True


def task_invariants_report():
    """The transcription cross-check: recompute D, T, per-operand transfer
    bytes and tasklet count from each candidate's compiled artifact and
    print them next to the values the transcriber derived from the trace.
    A disagreement means the transcription (or its reading of the trace)
    is wrong -- before any hardware time is spent on it."""
    for source in POINT_SOURCES:
        for bench in WORKLOADS:
            if not (POINTS_DIR / source / f"{bench}.json").exists():
                continue
            yield {
                "name": f"{source}:{bench}",
                "actions": [(_invariants_report, [source, bench])],
                "uptodate": [False],
            }
