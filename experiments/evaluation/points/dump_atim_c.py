"""Generate the DPU C that ATiM's schedules compile to.

    python points/dump_atim_c.py [--traces points/traces]
                                 [--only gemv_1024_1_1024.reproduced ...]

Writes `points/traces/{stem}.dpu.c` beside every `{stem}.tir.py` dump: the
kernel source ATiM's UPMEM backend emits for that schedule, which is what a
transcription in `points/{atim_*}/` has to be checked against.

The TIR dump alone does not answer whether a transcription is faithful.
Two schedules with the same tiling factors still differ in what each DPU is
sent, how many MRAM transfers a tile costs, and where the partial sums are
written back -- none of which is legible in the TIR but all of which is
plain in the generated C. Comparing our `{fn}.dpu.c` against ATiM's is the
check; the tiling factors agreeing is not.

This is a rerun of ATiM's own code generation, not a reimplementation: the
dump is parsed back into the module it printed, and `tvm.build` is called on
it exactly as `evaluation/base.py:pre_kernel` does. The optimization level
rides on the module's own root-block annotation (`storage_flatten.cc` reads
it), so the pass config is the neutral one and the schedule decides.

Beware ATiM's `logs/results_*/<lambda>_0/upmem.c`: it is whatever schedule
that tuning run happened to build last, unlabelled, and is not necessarily
the measured incumbent. This script names its output after the trace.

Running it needs ATiM's TVM and the UPMEM SDK, but no DPUs; `_atim_env`
finds both and re-execs under an interpreter that can load them.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import _atim_env

HERE = pathlib.Path(__file__).resolve().parent

# "gemv_1024_1_1024.reproduced.tir.py" -> "gemv_1024_1_1024.reproduced"
TIR_SUFFIX = ".tir.py"


def build_c(mod) -> str:
    """The DPU kernel source, generated the way ATiM generates it.

    `evaluation/base.py:pre_kernel` builds under exactly these two config
    keys and reads the source off the imported (device) module; `-1` is its
    default optimization level, meaning "whatever the schedule annotated".
    """
    import tvm
    from tvm.target import Target

    config = {"tir.UpmemUseDummyKernel": False, "tir.UpmemKernelOptimize": -1}
    with tvm.transform.PassContext(config=config):
        func = tvm.build(mod, target=Target("upmem"), name="kernel")
    return func.imported_modules[0].get_source()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=pathlib.Path, default=HERE / "traces")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        help="trace stems to generate, e.g. gemv_1024_1_1024.reproduced",
    )
    _atim_env.add_arguments(parser)
    args = parser.parse_args()

    _atim_env.bootstrap(__file__, args.atim, args.python)
    _atim_env.warn_without_sdk()

    only = set(args.only or ())
    unmatched = set(only)
    written, problems = 0, []

    for tir in sorted(args.traces.glob(f"*{TIR_SUFFIX}")):
        stem = tir.name[: -len(TIR_SUFFIX)]
        if only and stem not in only:
            continue
        unmatched.discard(stem)
        out = tir.with_name(f"{stem}.dpu.c")
        try:
            source = build_c(_atim_env.load_module(tir))
        except Exception as exc:
            # One schedule failing to build says nothing about the others, and
            # the whole point is to get as many side-by-sides as possible.
            problems.append(f"{stem}: {type(exc).__name__}: {exc}".replace("\n", " "))
            continue
        header = (
            f"// DPU kernel ATiM's UPMEM backend generates for {tir.name}.\n"
            f"// Regenerate with `python points/dump_atim_c.py --only {stem}`.\n"
        )
        out.write_text(header + source.rstrip("\n") + "\n")
        written += 1
        print(f"{out.relative_to(args.traces.parent.parent)}")

    problems += [
        f"{stem}: --only names it, but no such trace" for stem in sorted(unmatched)
    ]
    print(f"\n{written} kernels written")
    if problems:
        print(f"\n{len(problems)} could not be generated:")
        for problem in problems:
            print(f"  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
