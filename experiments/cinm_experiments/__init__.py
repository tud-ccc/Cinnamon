"""Reusable building blocks for cinm-mlir experiments: invoking cinm-opt
(exhaustive / BO / single-config eval), compiling+running on real UPMEM
hardware, parallelizing that work, and turning raw benchmark output into
net-time measurements or per-source aggregate CSVs.

Experiments (e.g. experiments/cinm1comparison/dodo.py) import this package
and define their pipeline as plain Python function calls / doit tasks; there
is no CLI or Makefile glue layer between the steps.
"""

from .prims import PRIMS, Prim  # noqa: F401

# The prim_<name>.mlir stems, for pipelines that key off source-module names
# rather than Prim objects. Derived from PRIMS so there is one list of
# primitives, not two.
ALL_PRIMS = tuple(f"prim_{name}" for name in PRIMS)
