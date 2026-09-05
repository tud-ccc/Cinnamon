"""results/campaign.csv + campaign_ref.csv -- the search-strategy
comparison's raw anytime curves (docs/SearchStrategyPlan.md).

campaign.csv holds one row per budget-consuming evaluation of every
(arm, function, seed): eval_idx is the evaluation's rank within its seed
(init sample included), best_so_far the running minimum over finite
costs, elapsed_ms/cpu_ms the seed's cumulative clocks at that evaluation
(timings.csv merged by rank; empty where the counts disagree).

Only campaign_* dumps are read -- never data/*/search or data/*/sample.
Those predate the 2026-08-20 lowering changes, which moved the simulated
cost of identical configurations by up to 2x, so folding them into the
curves or the reference would mix two cost landscapes (see
docs/SearchStrategyPlan.md's progress log). The `bananas` control is
therefore a campaign arm of its own, re-run on the current lowering.

campaign_ref.csv holds the per-function reference optimum: the pooled
minimum over every arm's observations. Not a true optimum but a lower
envelope that tightens as arms run; it is recorded so regret can be
recomputed when it moves. Missing-tolerant like every assemble step:
arms not yet run simply contribute no rows.
"""

from __future__ import annotations

import csv
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

# Arm -> dump glob (relative to data/). Keep in sync with dodo.py's
# CAMPAIGN_ARMS.
ARM_DUMPS = {
    "bananas": "*/campaign_bananas/dump",
    "bananas_rand": "*/campaign_bananas_rand/dump",
    "bananas_rand4k": "*/campaign_bananas_rand4k/dump",
    "random": "*/campaign_random/dump",
    "descent": "*/campaign_descent/dump",
    "ga": "*/campaign_ga/dump",
}

# pool.csv columns that are not search-space dimensions.
NON_DIM_COLS = {
    "visited",
    "valid",
    "cost",
    "eval_iter",
    "eval_time_ms",
    "cpu_time_ms",
    "mu",
    "sigma",
    "acq",
}


def _read_seed(pool_csv: pathlib.Path) -> list[tuple[int, float]]:
    """The seed's evaluations as (eval_iter, cost), one per budget-consuming
    evaluation (visited rows with a recorded cost; inf = timed-out sim)."""
    evals = []
    with open(pool_csv) as f:
        for row in csv.DictReader(f):
            if not row["cost"]:
                continue
            evals.append((int(row["eval_iter"]), float(row["cost"])))
    evals.sort()
    return evals


def _read_timings(seed_dir: pathlib.Path) -> list[tuple[float, float]]:
    """Cumulative (elapsed_ms, cpu_ms) per successful evaluation, in order."""
    path = seed_dir / "timings.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return [(float(r["elapsed_ms"]), float(r["cpu_ms"])) for r in csv.DictReader(f)]


def assemble() -> bool:
    rows = []
    ref: dict[str, float] = {}
    ref_n: dict[str, int] = {}

    def fold_ref(fn: str, cost: float) -> None:
        if math.isfinite(cost):
            ref[fn] = min(ref.get(fn, math.inf), cost)
            ref_n[fn] = ref_n.get(fn, 0) + 1

    for arm, pattern in ARM_DUMPS.items():
        for pool_csv in sorted(DATA_DIR.glob(f"{pattern}/infer_*/seed_*/pool.csv")):
            seed_dir = pool_csv.parent
            fn = seed_dir.parent.name.removeprefix("infer_")
            seed = int(seed_dir.name.removeprefix("seed_"))
            evals = _read_seed(pool_csv)
            timings = _read_timings(seed_dir)
            best = math.inf
            for idx, (_, cost) in enumerate(evals):
                fold_ref(fn, cost)
                if math.isfinite(cost):
                    best = min(best, cost)
                elapsed, cpu = timings[idx] if idx < len(timings) else ("", "")
                rows.append(
                    {
                        "arm": arm,
                        "fn": fn,
                        "seed": seed,
                        "eval_idx": idx,
                        "cost": cost,
                        "best_so_far": best if math.isfinite(best) else "",
                        "elapsed_ms": elapsed,
                        "cpu_ms": cpu,
                    }
                )

    if not rows:
        print("assemble_campaign: no campaign dumps present yet, nothing to assemble")
        return True

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "campaign.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS_DIR / "campaign_ref.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fn", "ref_cost", "n_pooled"])
        writer.writeheader()
        for fn in sorted(ref):
            writer.writerow({"fn": fn, "ref_cost": ref[fn], "n_pooled": ref_n[fn]})
    arms = sorted({r["arm"] for r in rows})
    print(
        f"assemble_campaign: {len(rows)} rows, arms={arms}, "
        f"{len(ref)} reference optima -> {RESULTS_DIR / 'campaign.csv'}"
    )
    return True


if __name__ == "__main__":
    assemble()
