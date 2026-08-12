"""Reusable doit building blocks for compile-and-bench pipelines.

Factored out of cinm1comparison/dodo.py so that every experiment dodo (the
CINM1 comparison, the paper evaluation in experiments/evaluation/) drives
the same compile/bench/retry machinery instead of copying it. This module
holds the pieces that are the same whatever the experiment measures:

- MeasureRoots: the on-disk layout of one compile+run campaign, keyed by a
  (system, fn_name, label) config identity (compile_run.Config).
- compile_one / compile_best: fallible-per-config compile actions.
- bench_one_config: the always-touch-the-marker hardware bench action.
- BenchChain: chains bench tasks into ONE strict sequence, across task
  creators, so concurrent hardware runs never skew wall-clock timing.
- clear_failed_compiles / clear_failed_bench: the retry-task actions.

The dodo keeps what is experiment-specific: which configs exist, in which
order they bench, and what is compared/plotted afterwards. Everything here
communicates through files on disk (markers, output dirs), matching how
doit connects stages.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil

from . import compile_run, measurements, pools


# ── on-disk layout ───────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MeasureRoots:
    """Where one campaign's compiles and runs live. Everything below is
    derived from a config's (system, fn_name, label) identity -- never from
    its params/lower, which may not be known when paths are needed (doit
    wires tasks by file, and a task's targets must be known at creation).

    cinm1comparison instantiates one of these per prim
    (data/<prim>/compiled, data/<prim>/run); the evaluation pipeline one per
    (benchmark, stack). The layout below the two roots is fixed:
      compile_root/<system>/<fn_name>/<label>/{config.csv,lowered.mlir,bin/,compile.done}
      run_root/<system>/<fn_name>/<label>/{output/,bench.done}
    """

    compile_root: pathlib.Path
    run_root: pathlib.Path

    # compile side
    def config_dir(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        return self.compile_root / system / fn_name / label

    def compile_marker(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        return self.config_dir(system, fn_name, label) / "compile.done"

    def bench_bin(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        return self.config_dir(system, fn_name, label) / "bin" / f"bench_{fn_name}"

    # run side
    def run_config_dir(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        return self.run_root / system / fn_name / label

    def run_output_dir(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        return self.run_config_dir(system, fn_name, label) / "output"

    def bench_marker(self, system: str, fn_name: str, label: str) -> pathlib.Path:
        """Per-config bench-attempted marker (touched whether the run
        succeeded or not -- see bench_one_config), sibling of that config's
        output/ dir. Lets doit save bench progress config-by-config instead
        of only per-group, so an interrupted bench resumes where it left
        off."""
        return self.run_config_dir(system, fn_name, label) / "bench.done"

    # Config-taking conveniences.
    def _id(self, c: compile_run.Config) -> tuple[str, str, str]:
        return (c.system, c.fn_name, c.label)

    def compile_marker_of(self, c: compile_run.Config) -> pathlib.Path:
        return self.compile_marker(*self._id(c))

    def bench_bin_of(self, c: compile_run.Config) -> pathlib.Path:
        return self.bench_bin(*self._id(c))

    def run_output_dir_of(self, c: compile_run.Config) -> pathlib.Path:
        return self.run_output_dir(*self._id(c))

    def bench_marker_of(self, c: compile_run.Config) -> pathlib.Path:
        return self.bench_marker(*self._id(c))


# ── compile actions (fallible per config, never raise) ──────────────────────


def compile_one(
    config: compile_run.Config,
    roots: MeasureRoots,
    marker: pathlib.Path | None = None,
) -> bool:
    """Never raises: a config that fails to compile is recorded (printed +
    left out of the marker's sibling bin/) but must not block sibling
    configs' bench task from running -- doit treats a raised exception as a
    hard failure and skips every downstream task that depends on it, which
    is more than we want for one bad config out of many. discover_compiled()
    already treats a missing bench_* binary as a per-config failure, so
    downstream stages tolerate this fine."""
    marker = marker or roots.compile_marker_of(config)
    compiled = compile_run.compile_config(config, compile_root=roots.compile_root)
    if not compiled.ok:
        print(
            f"  FAIL compile: {config.system} {config.fn_name} {config.label}: {compiled.error}"
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return True


def compile_best(
    config: compile_run.Config,
    pool_csv: pathlib.Path,
    roots: MeasureRoots,
    marker: pathlib.Path | None = None,
) -> bool:
    """compile_one, with config.params filled from the best row of a search's
    pool.csv first. Returns False (no marker) when the pool has no valid row
    -- the search found nothing, which IS a hard failure worth surfacing."""
    config.params = pools.best_in_pool(pool_csv)
    if not config.params:
        return False
    return compile_one(config, roots, marker)


# ── bench action (sequential hardware runs) ─────────────────────────────────


def bench_one_config(
    config: compile_run.Config,
    roots: MeasureRoots,
    *,
    iters: int,
    bench_marker: pathlib.Path | None = None,
) -> bool:
    """Never raises, like compile_one: a config whose compile failed
    (compile.done marker present, no bench_* binary) is skipped rather than
    treated as a hard doit failure, so it doesn't block sibling configs'
    bench tasks. The bench.done marker is always touched, even when the
    hardware run itself fails, so a flaky config doesn't get retried on
    every doit invocation -- rerun it explicitly via the retry task
    (clear_failed_bench)."""
    bench_marker = bench_marker or roots.bench_marker_of(config)
    compiled = compile_run.discover_compiled([config], compile_root=roots.compile_root)[
        0
    ]
    if not compiled.ok:
        print(
            f"  SKIP bench (not compiled): {config.system} {config.fn_name} {config.label}"
        )
    else:
        r = compile_run.run_config(compiled, run_root=roots.run_root, iters=iters)
        if not r.ok and compile_run.is_dpu_allocation_error(r.error):
            # retry once: allocation races with whatever else holds ranks
            r = compile_run.run_config(compiled, run_root=roots.run_root, iters=iters)
        if not r.ok:
            print(
                f"  FAIL run: {config.system} {config.fn_name} {config.label}: {r.error[:200]}"
            )
    bench_marker.parent.mkdir(parents=True, exist_ok=True)
    bench_marker.touch()
    return True


# ── the global bench chain ───────────────────────────────────────────────────


class BenchChain:
    """Chains bench tasks -- which, unlike compile, must run strictly one at
    a time so concurrent hardware runs don't skew wall-clock timing -- into
    ONE sequence spanning every group of tasks, even when they are generated
    by different task creator functions (doit doesn't allow two creators to
    share a basename, so there is no single 'bench' task group to chain
    within).

    Usage: build the chain ONCE, in the full desired bench order, from every
    bench task's fully qualified name ("basename:name"); then each task's
    task_dep is prev_of(its own name). The order of registration is the
    order of execution.

    Must be task_dep, not file_dep on the predecessor's bench.done marker:
    doit's implicit file_dep -> task_dep inference is computed once, when
    each delayed creator's tasks are generated, against whatever targets are
    already known at that moment -- it does NOT retroactively wire up a
    file_dep against a target a different, not-yet-expanded delayed creator
    produces later. Naming the task directly resolves correctly instead."""

    def __init__(self):
        self._order: list[str] = []
        self._pos: dict[str, int] = {}

    def register(self, task_full_name: str) -> None:
        if task_full_name in self._pos:
            raise ValueError(f"bench task registered twice: {task_full_name}")
        self._pos[task_full_name] = len(self._order)
        self._order.append(task_full_name)

    def register_all(self, task_full_names) -> None:
        for name in task_full_names:
            self.register(name)

    def prev_of(self, task_full_name: str) -> list[str]:
        """The task_dep list for this bench task: its predecessor in the
        chain, or nothing for the first task overall."""
        i = self._pos[task_full_name]
        return [] if i == 0 else [self._order[i - 1]]


# ── retry actions ────────────────────────────────────────────────────────────


def clear_failed_compiles(configs, roots: MeasureRoots) -> int:
    """Delete the compile output of every config whose compile.done marker
    exists but whose bench_* binary doesn't -- i.e. every config compile_one
    recorded as failed instead of leaving broken. With the marker gone, the
    next doit run sees a missing target and retries just those configs;
    configs that already compiled are untouched. Returns how many were
    cleared."""
    n = 0
    for c in configs:
        marker = roots.compile_marker_of(c)
        bench_bin = roots.bench_bin_of(c)
        if marker.exists() and not bench_bin.exists():
            print(f"  retry: {c.system} {c.fn_name} {c.label}")
            shutil.rmtree(marker.parent)
            n += 1
    print(f"cleared {n} failed compile(s)")
    return n


def clear_failed_bench(configs, roots: MeasureRoots) -> int:
    """Delete the bench.done marker of every config whose compile succeeded
    but whose run didn't leave a measurable result (measurements.net_time_ms
    is None) -- i.e. every config that failed on hardware instead of just
    being un-benched yet. With the marker gone, the next doit bench run
    retries just those configs. Returns how many were cleared."""
    n = 0
    for c in configs:
        bin_path = roots.bench_bin_of(c)
        bench_marker = roots.bench_marker_of(c)
        if not (bin_path.exists() and bench_marker.exists()):
            continue
        output_dir = roots.run_output_dir_of(c)
        if measurements.net_time_ms(output_dir) is None:
            print(f"  retry bench: {c.system} {c.fn_name} {c.label}")
            bench_marker.unlink()
            n += 1
    print(f"cleared {n} failed bench(es)")
    return n
