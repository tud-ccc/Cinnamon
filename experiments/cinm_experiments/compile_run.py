"""Compile one fixed configuration down to a real UPMEM binary (via
Config.lower -- either cinmopt.eval_solution_lowerer for CINM 2.0 or
cinm1.lowerer for CINM 1.0 -- + this package's own bench-single Makefile
target) and benchmark it on hardware."""

from __future__ import annotations

import csv
import dataclasses
import pathlib
import shlex
import subprocess
from typing import Callable

from tqdm import tqdm

from . import parallel
from .paths import COMPILE_MAKEFILE_DIR


@dataclasses.dataclass
class Config:
    system: str  # caller-defined tag, e.g. "cinm1" / "cinm2" -- keeps compile/run dirs apart
    fn_name: str
    label: str  # unique id within fn_name, e.g. a BO seed or "D8_T4"
    params: (
        dict  # dpus/tasklets (+ whatever else); informational, e.g. DPU-cap accounting
    )
    fn_module: pathlib.Path  # single-function source .mlir
    prim: str  # e.g. "gemv" / "red", passed as bench-single's PRIM=
    lower: Callable[
        [pathlib.Path, pathlib.Path, pathlib.Path], subprocess.CompletedProcess
    ]
    # lower(fn_module, out_file, log_file) -> CompletedProcess; produces
    # out_file at the "upmem dialect" stage bench-single expects as input.

    def dir(self, root: pathlib.Path):
        return root / self.system / self.fn_name / self.label


@dataclasses.dataclass
class CompiledConfig:
    config: Config
    compile_dir: pathlib.Path
    ok: bool
    error: str = ""

    @property
    def num_dpus(self) -> int:
        return int(self.config.params.get("dpus", 1))


@dataclasses.dataclass
class RunResult:
    compiled: CompiledConfig
    ok: bool
    output_dir: pathlib.Path
    error: str = ""


def compile_config(config: Config, *, compile_root: pathlib.Path) -> CompiledConfig:
    config_dir = (
        pathlib.Path(compile_root) / config.system / config.fn_name / config.label
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "config.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "fn_name", "label", *config.params.keys()]
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": config.system,
                "fn_name": config.fn_name,
                "label": config.label,
                **config.params,
            }
        )

    lowered = config_dir / "lowered.mlir"
    r = config.lower(
        config.fn_module, lowered, config_dir / "cinm-opt.log", **config.params
    )
    if r.returncode != 0:
        return CompiledConfig(
            config, config_dir, False, f"cinm-opt failed, see {config_dir}/cinm-opt.log"
        )

    ir_dir, bin_dir = config_dir / "ir", config_dir / "bin"
    cmd = [
        "make",
        "-C",
        str(COMPILE_MAKEFILE_DIR),
        f"SRC_MLIR={lowered.resolve()}",
        f"IR_DIR={ir_dir.resolve()}",
        f"BIN_DIR={bin_dir.resolve()}",
        f"BENCH_FN={config.fn_name}",
        f"PRIM={config.prim}",
        "bench-single",
    ]
    with open(config_dir / "make.sh", "w") as f:
        f.write(f"#!/bin/sh\n{shlex.join(cmd)}\n")

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        (config_dir / "make_stderr.txt").write_text(r.stderr)
        return CompiledConfig(
            config, config_dir, False, f"make failed:\n{r.stderr[-10000:]}"
        )

    return CompiledConfig(config, config_dir, True)


def run_config(
    compiled: CompiledConfig, *, run_root: pathlib.Path, iters: int
) -> RunResult:
    if not compiled.ok:
        return RunResult(compiled, False, pathlib.Path(), "not compiled")
    cfg = compiled.config
    bin_dir = compiled.compile_dir / "bin"
    bench_bin = bin_dir / f"bench_{cfg.fn_name}"
    output_dir = (
        pathlib.Path(run_root) / cfg.system / cfg.fn_name / cfg.label / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [str(bench_bin), str(output_dir), str(iters)],
        capture_output=True,
        text=True,
        cwd=str(bin_dir / cfg.fn_name),
    )
    if r.returncode != 0:
        # Persisted next to (not inside) output/ -- same config_dir level as
        # compile's own cinm-opt.log/make_stderr.txt -- so a failure survives
        # past this process for later triage (see cinm_experiments.failures),
        # not just the truncated in-memory RunResult.error below.
        (output_dir.parent / "error.txt").write_text(r.stderr)
        return RunResult(compiled, False, output_dir, r.stderr[-1000:])
    return RunResult(compiled, True, output_dir)


def discover_compiled(
    configs: list[Config], *, compile_root: pathlib.Path
) -> list[CompiledConfig]:
    """Reconstruct CompiledConfig entries for configs already compiled by a
    previous compile_configs() run, without recompiling -- for --run-only."""
    results = []
    for config in configs:
        config_dir = (
            pathlib.Path(compile_root) / config.system / config.fn_name / config.label
        )
        bench_bin = config_dir / "bin" / f"bench_{config.fn_name}"
        if bench_bin.exists():
            results.append(CompiledConfig(config, config_dir, True))
        else:
            results.append(
                CompiledConfig(
                    config,
                    config_dir,
                    False,
                    f"not compiled yet (expected {bench_bin})",
                )
            )
    return results


def is_dpu_allocation_error(error: str) -> bool:
    return "allocation error" in error.lower()


def compile_configs(
    configs: list[Config], *, compile_root: pathlib.Path, workers: int = 8
) -> list[CompiledConfig]:
    """Compile every config in parallel (threads -- the actual work happens
    in spawned cinm-opt/make subprocesses either way, and a lambda closure
    can't be pickled for ProcessPoolExecutor). Returns one CompiledConfig per
    input config, in no particular order."""
    compiled = parallel.run_parallel(
        configs,
        lambda c: compile_config(c, compile_root=compile_root),
        workers=workers,
        desc="compile",
        use_threads=True,
    )
    for c in compiled:
        if not c.ok:
            print(
                f"  FAIL compile: {c.config.system} {c.config.fn_name} {c.config.label}: {c.error}"
            )
    return compiled


def run_configs(
    compiled: list[CompiledConfig], *, run_root: pathlib.Path, iters: int = 10
) -> list[RunResult]:
    """Benchmark every compiled config one at a time -- sequential, not
    parallel, for accurate wall-clock timing (concurrent hardware runs
    contend for host/DPU resources and skew measurements). Configs that
    failed to compile are passed through as failed RunResults. Retries once,
    immediately, on a transient DPU allocation error."""
    results = []
    for c in tqdm(compiled, desc="run"):
        if not c.ok:
            results.append(RunResult(c, False, pathlib.Path(), c.error))
            continue
        r = run_config(c, run_root=run_root, iters=iters)
        if not r.ok and is_dpu_allocation_error(r.error):
            r = run_config(c, run_root=run_root, iters=iters)
        if not r.ok:
            print(
                f"  FAIL run: {c.config.system} {c.config.fn_name} {c.config.label}: {r.error[:200]}"
            )
        results.append(r)
    return results
