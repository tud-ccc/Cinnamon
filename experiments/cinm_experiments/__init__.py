"""Reusable building blocks for cinm-mlir experiments: invoking cinm-opt
(exhaustive / BO / single-config eval), compiling+running on real UPMEM
hardware, parallelizing that work, and turning raw benchmark output into
net-time measurements or per-source aggregate CSVs.

Experiments (e.g. experiments/cinm1comparison/dodo.py) import this package
and define their pipeline as plain Python function calls / doit tasks; there
is no CLI or Makefile glue layer between the steps.
"""

ALL_PRIMS = (
    "prim_gemv",
    "prim_geva",
    "prim_red",
    "prim_mmtv",
    "prim_mtv",
    "prim_ttv",
    "prim_va",
)
