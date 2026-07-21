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

from cinm_experiments import compile_run, cinm1, cinmopt  # noqa: E402


# cinmopt.bo_multiseed(
#     src="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_red/_split/red_256MB.mlir",
#     out_dir=HERE / "foodata",
#     debug=True,
#     nolog=True,
#     infer_opts={
#         "fixed-dpus": 3,
#         "fixed-tasklets": 2,
#         "simulator": "hybrid",
#         "eval-timeout-ms": 400
#     },
# )


# What i'm doing right now is trying to compare the exact same configuration compiled by CINM 1 and CINM 2. The only difference should be the CINM2 opts:
# - Moving gather out of the inner host loop
# - Sharing WRAM for the vector buffer
# - Static WRAM allocations instead of dynamic
# - On-DPU transfers orchestrated by tasklet 0.


# I expect maybe the synchronization overhead is the problem
# Maybe DPUs can do 2 transfers at the same time? In which case blocking for the full transfer is bad? Can the simulator help me with this?

# GEMV 64MB: 4096x4096

cinm2parms = {
    "dpus": 256,
    "tasklets": 4,
    "wramRow": 1,
    "wramCol": 1024,
    "dpuCols": 1,
    "mramRow": 4,
    "mramCol": 1024,
}
# {dpus=256, tasklets=4, wramRow=1, wramCol=1024, dpuCols=1, mramRow=4096, mramCol=1024}
conf = compile_run.Config(
    system="cinm2as1",
    fn_name="gemv_64MB",
    label="foo",
    params=cinm2parms,
    fn_module="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_gemv/_split/gemv_64MB.mlir",
    prim="gemv",
    lower=cinmopt.eval_solution_lowerer(params=cinm2parms),
)

compile_run.compile_config(conf, compile_root=HERE / "foodata")

conf = compile_run.Config(
    system="cinm1",
    fn_name="gemv_64MB",
    label="foo",
    params={"dpus": 256, "tasklets": 4},
    fn_module="/home/clement.fournier/Work/cinm-mlir/experiments/cinm1comparison/data/prim_gemv/_split/gemv_64MB.mlir",
    prim="gemv",
    lower=cinm1.lowerer(dpus=256, tasklets=4),
)

compile_run.compile_config(conf, compile_root=HERE / "foodata")

