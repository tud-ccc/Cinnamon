"""pool.csv utilities: loading and selecting configs."""

from __future__ import annotations

import pathlib

import pandas as pd

NON_PARAM_COLS = frozenset(
    {
        "visited",
        "valid",
        "cost",
        "eval_iter",
        "eval_time_ms",
        "cpu_time_ms",
        "mu",
        "sigma",
        "acq",
        "index",
    }
)


def param_cols(df: pd.DataFrame) -> list[str]:
    """The config-space dimension columns, in declaration order, as dumped by
    cinm-opt. eval_solution() needs values supplied in this exact order."""
    return [c for c in df.columns if c not in NON_PARAM_COLS]


def fn_dirs(root: pathlib.Path):
    """Yield (fn_name, dir) for every function subdirectory of root, whether
    or not it has an "infer_" dump-dir prefix."""
    for d in sorted(pathlib.Path(root).iterdir()):
        if d.is_dir():
            yield d.name.removeprefix("infer_"), d


def load_valid(pool_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(pool_csv)
    if "valid" in df.columns:
        df = df[df["valid"] == 1]
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()].copy()
    df["cost"] = df["cost"].astype(float)
    return df


def select_best(
    pool_csv: pathlib.Path, *, top_frac: float = 0.10, min_configs: int = 200
) -> tuple[pd.DataFrame, int, int]:
    """Rank every valid config in pool_csv by cost, keep the best
    max(top_frac * N, min_configs). Returns (kept_df, n_valid, n_kept)."""
    df = load_valid(pool_csv)
    if df.empty:
        return df, 0, 0
    n_valid = len(df)
    n_keep = max(int(n_valid * top_frac), min(min_configs, n_valid))
    top = df.sort_values("cost").head(n_keep)
    return top, n_valid, n_keep


def best_per_seed(results_dir: pathlib.Path):
    """Yield (fn_name, seed, params_dict) for the lowest-cost visited row of
    every {results_dir}/{fn_name}/seed_{N}/pool.csv (the output of
    cinmopt.bo_multiseed)."""
    for fn_name, fn_dir in fn_dirs(results_dir):
        for seed_dir in sorted(fn_dir.iterdir()):
            pool_csv = seed_dir / "pool.csv"
            best_conf = best_in_pool(pool_csv)
            if not best_conf:
                continue
            seed = seed_dir.name.removeprefix("seed_")
            yield fn_name, seed, best_conf


def best_in_pool(pool_csv: pathlib.Path):
    """The lowest-cost visited row of pool_csv as a params dict (the form
    eval_solution() consumes), or None if the pool is missing or has no
    valid visited row."""
    if not pool_csv.exists():
        return None
    df = pd.read_csv(pool_csv)
    if "visited" in df.columns:
        df = df[df["visited"] == 1]
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()]
    if df.empty:
        return None
    cols = param_cols(df)
    row = df.loc[df["cost"].astype(float).idxmin()]
    return {c: int(row[c]) for c in cols}


def _rows_to_params(df: pd.DataFrame) -> list[dict]:
    cols = param_cols(df)
    return [{c: int(row[c]) for c in cols} for _, row in df.iterrows()]


def top_k(pool_csv: pathlib.Path, k: int) -> list[dict]:
    """The k lowest-predicted-cost valid configs of pool_csv, each as a
    params dict for eval_solution() -- the evaluation pipeline's B3 block
    (best-of-space by the cost model; docs/EvaluationImplementationPlan.md).
    Rows whose cost is not a number (never evaluated / timed out) are
    excluded, which for a full exhaustive pool means only configs the
    simulator could price compete for the top."""
    df = load_valid(pool_csv)
    if df.empty:
        return []
    return _rows_to_params(df.sort_values("cost").head(k))


def sample_rows(pool_csv: pathlib.Path) -> list[dict]:
    """Every visited config of pool_csv as a params dict for
    eval_solution(), in row order -- the evaluation pipeline's B1->B2 glue
    (the uniform sample's pool.csv only contains the sampled rows unless
    dump-full-pool was forced on, and only visited ones carry a cost)."""
    if not pool_csv.exists():
        return []
    df = pd.read_csv(pool_csv)
    if "visited" in df.columns:
        df = df[df["visited"] == 1]
    return _rows_to_params(df)
