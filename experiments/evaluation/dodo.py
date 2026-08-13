"""doit tasks for the paper evaluation (§8 of the paper).

This pipeline collects the measurements behind the paper's §8: the shared
uniform sample (B1), the space dumps the transcription workflow reads (B0),
and -- as later phases land -- the top-k, search, transcription-point,
cinm1-sweep and RQ4 stacks. Structure and conventions follow
cinm1comparison/dodo.py: stages connected by files, fallible-per-config
compiles, ONE strict hardware-bench chain, retry tasks. The shared
machinery lives in cinm_experiments.doit_blocks.

Phase 0 scope (this file today): B0 `space` + B1 `sample` -> compile ->
bench, with retries; the A2 probe stack; the B6 assemble layer and the
plot/table scripts it feeds.

Usage:
  doit space             # dump every benchmark's space.json (no simulator)
  doit sample            # draw the shared uniform sample (simulator only)
  doit                   # everything up to bench_sample (hardware!)
  doit retry_failed_compiles && doit
  doit retry_failed_bench && doit bench_sample
"""

from __future__ import annotations

import pathlib
import sys

from doit import create_after
from doit.reporter import ProgressBarReporter

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))  # sibling modules (assemble) under doit

from cinm_experiments import cinmopt, compile_run, doit_blocks, pools, ALL_PRIMS  # noqa: E402
from cinm_experiments import space as space_mod  # noqa: E402
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
    k_top=200,  # B3 (not yet wired): measured ground truth for top-k overlap
    n_seeds=32,  # B4 (not yet wired): matches the cinm1comparison campaign
    iters=6,  # measurement repetitions per hardware run
    # The hybrid model's switching point, same value as the search stack
    # (cinm1comparison): cycle-accurate within the budget, fast model past
    # it. Part of the model's definition, not a performance knob -- B1's
    # predicted costs and every search must use the same value.
    eval_timeout_ms=400,
    # A2's Cartesian draw. Same rule-of-three arithmetic as n_sample: with
    # feasible densities around 1e-10 essentially every draw lands in the
    # rejected region, so 0-lowers-of-300 bounds the false-negative rate of
    # the constraint system at 1% (95%). Its own seed: the draw is over the
    # Cartesian domains, not the feasible set, and must not be conflated
    # with the shared sample.
    n_a2=300,
    a2_seed=1848,
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


def sample_roots(bench: str) -> doit_blocks.MeasureRoots:
    return doit_blocks.MeasureRoots(
        compile_root=sample_dir(bench) / "compiled",
        run_root=sample_dir(bench) / "run",
    )


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


def _dump_space(bench: str) -> bool:
    cinmopt.dump_space(source_mlir(bench), space_dir(bench))
    return True


def task_space():
    """B0: dump every function's space.json -- sizes for tab:sufficiency,
    per-param docs + permutation tables for the manual ATiM transcription
    workflow. No simulator runs; safe anywhere."""
    for bench in WORKLOADS:
        yield {
            "name": bench,
            "file_dep": [str(source_mlir(bench))],
            "targets": [
                str(space_json(bench, fn)) for fn in list_functions(source_mlir(bench))
            ],
            "actions": [(_dump_space, [bench])],
        }


# ── B1: the shared uniform sample ────────────────────────────────────────────


def _draw_sample(bench: str) -> bool:
    cinmopt.random_sample(
        source_mlir(bench),
        sample_dir(bench),
        n_samples=OPTS["n_sample"],
        seed=OPTS["sample_seed"],
        infer_opts={
            # EXACTLY the model the search stack consumes -- simulator and
            # timeout together, because they are one model: hybrid runs
            # cycle-accurate under eval-timeout-ms and answers with the fast
            # model when it fires (UpmemPythonSimulator.cpp, SimMode::HYBRID)
            # -- with no timeout, "hybrid" is just cycle-accurate. RQ3's
            # fidelity claim is about what the flow chooses with, so the
            # sample's predicted costs must come from the same pairing the
            # search uses. The timeout does NOT censor the sample: a
            # timed-out config is priced by the fallback, not dropped.
            "simulator": "hybrid",
            "eval-timeout-ms": OPTS["eval_timeout_ms"],
            # What WOULD censor the sample is the cost cap, which rejects
            # and resamples: off, or the E1 percentile (rule of three needs
            # a uniform draw) and RQ3's rank correlation are void.
            "sample-max-cost-ms": 0,
        },
    )
    return True


def task_sample():
    """B1: draw the shared uniform sample -- ONE set of n_sample configs per
    function, drawn uniformly at random from the enumerated feasible set
    with a fixed seed, predicted costs recorded. Serves E1(b), RQ3's
    fidelity ground truth, §2's best-vs-median, and A2's accepted-but-fails
    check all at once; nothing downstream may redraw it."""
    for bench in WORKLOADS:
        yield {
            "name": bench,
            "file_dep": [str(source_mlir(bench))],
            "targets": [
                str(sample_pool_csv(bench, fn))
                for fn in list_functions(source_mlir(bench))
            ],
            "actions": [(_draw_sample, [bench])],
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


def _bench_task_name(bench: str, fn_name: str, label: str) -> str:
    return f"bench_sample:{bench}:{fn_name}:{label}"


@create_after(executed="sample", creates=["compile_sample", "bench_sample"])
def task_compile_sample():
    """Compile + bench every sampled config. Compiles are fallible per
    config; benches form ONE strict chain across all benchmarks (hardware
    timing, see doit_blocks.BenchChain). An eval-solution compile failure
    here is A2 DATA (accepted-but-fails must be 0), not just noise -- keep
    the logs."""
    chain = doit_blocks.BenchChain()
    per_bench: list[tuple[str, compile_run.Config]] = []
    for bench in WORKLOADS:
        for config in _sample_configs(bench):
            per_bench.append((bench, config))
            chain.register(_bench_task_name(bench, config.fn_name, config.label))

    for bench, config in per_bench:
        roots = sample_roots(bench)
        marker = roots.compile_marker_of(config)
        pool_csv = sample_pool_csv(bench, config.fn_name)
        yield {
            "basename": "compile_sample",
            "name": f"{bench}:{config.fn_name}:{config.label}",
            "file_dep": [str(pool_csv), str(config.fn_module)],
            "targets": [str(marker)],
            "actions": [(doit_blocks.compile_one, [config, roots, marker])],
        }

        bench_marker = roots.bench_marker_of(config)
        yield {
            "basename": "bench_sample",
            "name": f"{bench}:{config.fn_name}:{config.label}",
            # pool.csv too: a redrawn sample (parameter change) must
            # invalidate the measurements, not silently re-attribute them.
            "file_dep": [str(marker), str(pool_csv)],
            "task_dep": chain.prev_of(
                _bench_task_name(bench, config.fn_name, config.label)
            ),
            "targets": [str(bench_marker)],
            "actions": [
                (
                    doit_blocks.bench_one_config,
                    [config, roots],
                    dict(iters=OPTS["iters"], bench_marker=bench_marker),
                )
            ],
        }


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


# ── B6: assemble results/*.csv from whatever exists ─────────────────────────

import assemble  # noqa: E402  (sibling module; HERE is on sys.path via doit)


def _pair(roots: doit_blocks.MeasureRoots) -> tuple[pathlib.Path, pathlib.Path]:
    return (roots.compile_root, roots.run_root)


def _assemble_e1() -> bool:
    return assemble.assemble_e1(
        {
            bench: {
                "sample": _pair(sample_roots(bench)),
                "topk": _pair(topk_roots(bench)),
                "search": _pair(search_roots(bench)),
                "atim_transcribed": _pair(points_roots(bench, "atim")),
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
                "ours": _pair(search_roots(bench)),
                "cinm1": (
                    DATA_DIR / bench / "cinm1" / "compiled",
                    DATA_DIR / bench / "cinm1" / "run",
                ),
                "cinm1_rule": _pair(points_roots(bench, "cinm1rule")),
            }
            for bench in WORKLOADS
        },
        {system: OFFLINE_DIR / f"{system}.csv" for system in ("prim", "atim", "cpu")},
        RESULTS_DIR / "rq1.csv",
    )


def _assemble_rq2() -> bool:
    return assemble.assemble_rq2(
        {bench: search_dir(bench) for bench in WORKLOADS},
        {
            (bench, fn): space_json(bench, fn)
            for bench in WORKLOADS
            for fn in list_functions(source_mlir(bench))
        },
        RESULTS_DIR / "rq2.csv",
    )


def _assemble_rq3() -> bool:
    return assemble.assemble_rq3(
        {bench: _pair(sample_roots(bench)) for bench in WORKLOADS},
        RESULTS_DIR / "rq3.csv",
    )


def _assemble_a1() -> bool:
    return assemble.assemble_a1(
        {
            (bench, space): _pair(
                search_roots(bench)
                if space == "default"
                else ablate_roots(bench, space)
            )
            for bench in WORKLOADS
            for space in ["default", *ABLATE_SPACES]
        },
        RESULTS_DIR / "a1.csv",
    )


def _assemble_rq4() -> bool:
    return assemble.assemble_rq4(
        {
            (prog, arm): _pair(rq4_roots(prog, arm))
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
    ]:
        yield {
            "name": name,
            "actions": [action],
            "uptodate": [False],  # missing-tolerant: always re-derive
        }


PLOT_SCRIPTS = [
    "table_sufficiency.py",
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


# ── still to come ────────────────────────────────────────────────────────────
# B3 topk:    exhaustive predicted sweep (eval-timeout-ms=300) -> pools.top_k
#             -> same B2 shape as the sample stack.
# B4 search:  bo_multiseed(n_seeds) x {default, ablated spaces, simulators};
#             timings.csv feeds RQ2.
# B5 points:  points/atim/{bench}.json + points/cinm1rule/{bench}.json ->
#             eval-solution runs + invariants report.
# cinm1:      the (D,T) sweep with coverage accounting.
# assemble/plot: results/*.csv, missing-tolerant.
