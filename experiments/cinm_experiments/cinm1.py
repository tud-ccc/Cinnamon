"""CINM 1.0 compilation flow.

CINM 2.0's --upmem-infer-accelerator picks tile sizes by searching; CINM 1.0
has no search -- --cinm-infer-tile-sizes picks them with a plain
deterministic pass, so giving it a fixed (dpus, tasklets) working group is
enough to produce the one program CINM 1.0 generates. Compiling a CINM 1.0
program is then just a matter of calling the right passes in the right
order:

  1. --cinm-assign-platforms wraps the raw ops into a cinm.compute, but only
     sets a `cinm.available_platforms = [#foo]` list attribute, not a
     committed platform/accelerator -- to_cinm1_accelerator() gets there with
     a text substitution on a copy of the file, injecting a fixed CINM 1.0
     accelerator attribute (`on accelerator #upmem.array<1xDxT, #foo>`)
     directly.
  2. --cinm-infer-tile-sizes (+ tiling + isolate/deisolate) then picks tile
     sizes deterministically for that fixed working group.
  3. The same CINM -> CNM -> UPMEM lowering used everywhere in this repo
     carries the result to the "upmem dialect" stage that
     reduce_cost/Makefile's bench-single target expects as input.

Steps 2-3 are transcribed from testbench/Makefile (the original CINM 1.0
flow), stopping at its `.6.upmem.mlir` stage instead of continuing on to
DPU-C/host-LLVM as that Makefile itself does -- reduce_cost/Makefile's
bench-single already does that part, and is what compile_run.py drives.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

from .paths import DEFAULT_CINM_OPT
from .cinmopt import _infer_opts_str

PRE_PASSES = ["--cinm-assign-platforms"]

_AVAILABLE_PLATFORMS_RE = re.compile(r"cinm\.available_platforms\s*=\s*\[(#\w+)")
_COMPUTE_RE = re.compile(r"\bcinm\.compute\b")


def to_cinm1_accelerator(text: str, dpus: int, tasklets: int) -> str:
    """Replace `cinm.available_platforms = [#foo]` (--cinm-assign-platforms'
    output -- a list of candidates, not a committed platform/accelerator)
    with a fixed CINM 1.0 accelerator attribute directly on the op:
    `cinm.compute on accelerator #upmem.array<1xDxT, #foo>`. Assumes a single
    candidate platform, which always holds in this codebase."""
    m = _AVAILABLE_PLATFORMS_RE.search(text)
    if not m:
        raise RuntimeError("no cinm.available_platforms attribute found")
    platform = m.group(1)
    accel = f"on accelerator #upmem.array<1x{dpus}x{tasklets}, {platform}>"
    replaced, n = _COMPUTE_RE.subn(f"cinm.compute {accel}", text)
    if n == 0:
        raise RuntimeError("no cinm.compute op found")
    return replaced


# Transcribed from testbench/Makefile. Steps 1a/1b infer + apply tile sizes
# deterministically (no search); steps 2-6 are the ordinary CINM -> CNM ->
# UPMEM lowering, the same one that also carries CINM 2.0's own
# eval-solution output to this stage.
_STEP1A_PIPELINE = "builtin.module(func.func(cinm-infer-tile-sizes,cinm-tiling,cinm-isolate-compute-blocks,canonicalize))"
_STEP1B_PIPELINE = (
    "builtin.module(**cinm.compute_block(affine-loop-unroll{unroll-full-threshold=1},"
    "canonicalize),cinm-deisolate-compute-blocks)"
)

_STEP2 = [
    "--cinm-deisolate-compute-blocks",
    "--convert-cinm-to-cnm",
    "--canonicalize",
    "--cnm-hoist-workgroups",
    "--canonicalize",
    "--cse",
]
_STEP3 = [
    "--eliminate-empty-tensors",
    "--cse",
    "--one-shot-bufferize=bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map",
    "--cse",
    "--canonicalize",
    "--convert-linalg-to-affine-loops",
    "--buffer-loop-hoisting",
    "--buffer-hoisting",
    "--canonicalize",
    "--cse",
    "--buffer-results-to-out-params=hoist-static-allocs",
    "--canonicalize",
    "--cse",
]
_STEP4 = [
    "--promote-buffers-to-stack",
    "--fold-memref-alias-ops",
    "--canonicalize",
    "--affine-loop-fusion",
    "--sroa",
    "--canonicalize",
    "--affine-scalrep",
    "--loop-invariant-code-motion",
    "--affine-loop-invariant-code-motion",
    "--sroa",
    "--affine-scalrep",
    "--canonicalize",
    "--cse",
]


def _step5(use_upmem_scatter_api: bool):
    step = ["--lower-affine"]
    if not use_upmem_scatter_api:
        step.append("--cnm-ensure-scatter-gather-contiguous")
    step.extend(
        ["--buffer-loop-hoisting", "--buffer-hoisting", "--canonicalize", "--cse"]
    )
    return step


def _step6(use_upmem_scatter_api: bool):
    options = {
        "cinm1-codegen": "true",
        "use-sg-xfer-codegen": str(use_upmem_scatter_api).lower(),
    }
    pass_opts = _infer_opts_str(options)

    return [
        f"--convert-cnm-to-upmem={pass_opts}",
        "--cse",
        "--buffer-loop-hoisting",
        "--buffer-deallocation-pipeline",
        "--upmem-dedup-kernels",
        "--cse",
    ]


def _stages(use_upmem_scatter_api: bool):
    return [
        (["--pass-pipeline=" + _STEP1A_PIPELINE], "1a.mlir"),
        # (["--pass-pipeline=" + _STEP1B_PIPELINE], "1b.mlir"),
        (_STEP2, "2.mlir"),
        (_STEP3, "3.mlir"),
        (_STEP4, "4.mlir"),
        (_step5(use_upmem_scatter_api), "5.mlir"),
        (_step6(use_upmem_scatter_api), "6.mlir"),
    ]


def _run_stage(
    cinm_opt: pathlib.Path,
    in_file: pathlib.Path,
    flags: list[str],
    out_file: pathlib.Path,
    log,
) -> subprocess.CompletedProcess:
    cmd = [
        str(cinm_opt),
        str(in_file),
        *flags,
        "--mlir-print-ir-after-failure",
        "--mlir-print-assume-verified",
        "-o",
        str(out_file),
    ]
    log.write(" ".join(cmd) + "\n")
    return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


def compile_cinm1(
    fn_module: pathlib.Path,
    dpus: int,
    tasklets: int,
    out_file: pathlib.Path,
    *,
    work_dir: pathlib.Path,
    cinm_opt: pathlib.Path = DEFAULT_CINM_OPT,
    log_file: pathlib.Path | None = None,
    use_upmem_scatter_api: bool = False,
) -> subprocess.CompletedProcess:
    """Compile fn_module (a single-function module produced by
    split_source.split_source) as CINM 1.0 would for the fixed (dpus,
    tasklets) working group -- no search. Intermediate stage files are kept
    in work_dir for debugging (mirrors testbench/Makefile's own
    $(IR_PREFIX).N.*.mlir convention)."""
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file or (work_dir / "cinm-opt.log")

    with open(log_file, "w") as log:
        assigned = work_dir / "0.assigned.mlir"
        r = _run_stage(cinm_opt, fn_module, PRE_PASSES, assigned, log)
        if r.returncode != 0:
            return r

        pinned = work_dir / "0.pinned.mlir"
        pinned.write_text(to_cinm1_accelerator(assigned.read_text(), dpus, tasklets))

        stage = pinned
        for flags, name in _stages(use_upmem_scatter_api):
            nxt = work_dir / name
            r = _run_stage(cinm_opt, stage, flags, nxt, log)
            if r.returncode != 0:
                return r
            stage = nxt

    pathlib.Path(out_file).write_text(stage.read_text())
    return r


def lowerer(
    dpus: int, tasklets: int, *, cinm_opt: pathlib.Path = DEFAULT_CINM_OPT, **kwargs
):
    """A (fn_module, out_file, log_file) -> CompletedProcess callable, for use
    as compile_run.Config.lower -- compiles CINM 1.0's program for this fixed
    (dpus, tasklets) working group."""

    def _lower(
        fn_module: pathlib.Path, out_file: pathlib.Path, log_file: pathlib.Path
    ) -> subprocess.CompletedProcess:
        return compile_cinm1(
            fn_module,
            dpus,
            tasklets,
            out_file,
            work_dir=pathlib.Path(out_file).parent / "cinm1_stages",
            cinm_opt=cinm_opt,
            log_file=log_file,
            **kwargs,
        )

    return _lower
