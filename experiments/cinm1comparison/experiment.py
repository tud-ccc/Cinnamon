#!/usr/bin/env python3
"""Shared comparison logic for the CINM 1.0 vs CINM 2.0 codegen comparison
pipeline. The pipeline itself is driven by dodo.py (doit-based); this module
just holds compare(), which dodo.py's compare task imports.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from cinm_experiments import cinmopt  # noqa: E402


cinmopt.bo_multiseed(
    src="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_red/_split/red_256MB.mlir",
    out_dir=HERE / "foodata",
    debug=True,
    nolog=True,
    infer_opts={
        "fixed-dpus": 3,
        "fixed-tasklets": 2,
        "simulator": "hybrid",
        "eval-timeout-ms": 400
    },
)
