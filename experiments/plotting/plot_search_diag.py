#!/usr/bin/env python3
"""
Summarise the per-round search diagnostics a BO run writes next to pool.csv:
rounds.csv (what the acquisition selected and where the round's time went) and
batchdiag.csv (how clustered the top-q candidates were at each round).

Answers two questions the cost and regret curves cannot:

  * Would batching pay?  A round costs one surrogate fit and one evaluation.
    Evaluating q points per fit amortises the fit over q, so the ceiling on the
    speedup is set by the fit/evaluate time ratio -- which this reports.

  * Would a batch be redundant?  The top-q candidates under the acquisition are
    measured for clustering whether or not q of them were evaluated, so a run
    that selects one point per round still says what a q-wide batch would have
    covered.

Usage:
    python plot_search_diag.py RUNDIR [RUNDIR ...] -o OUTDIR

Each RUNDIR is a directory holding rounds.csv and batchdiag.csv (a seed_*/
directory of a dump-dir run). Multiple directories are overlaid.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH_META_COLS = {
    "round",
    "q",
    "q_eff",
    "q_eff_ref",
    "mean_pdist",
    "dispersion",
    "overlap_mu",
    "overlap_sigma",
}


# ── Loading ────────────────────────────────────────────────────────────────────
def load_run(rundir):
    rundir = Path(rundir)
    rounds_path = rundir / "rounds.csv"
    batch_path = rundir / "batchdiag.csv"
    if not rounds_path.exists():
        raise FileNotFoundError(f"{rounds_path} not found")
    rounds = pd.read_csv(rounds_path)
    batches = pd.read_csv(batch_path) if batch_path.exists() else None
    return rundir.name, rounds, batches


def load_budget(rundir, rounds):
    """End-to-end wall clock split by phase.

    Summed evaluation time is not wall clock: the LHS and validation phases
    evaluate concurrently across workers, so their total exceeds the elapsed
    time they occupy. Only the surrogate-guided phase is sequential, so its two
    components are taken from rounds.csv and the concurrent phases are what the
    elapsed time has left over.
    """
    rundir = Path(rundir)
    timings_path, space_path = rundir / "timings.csv", rundir / "space.json"
    if not timings_path.exists():
        return None
    total = pd.read_csv(timings_path)["elapsed_ms"].max() / 1000
    build = 0.0
    if space_path.exists():
        import json

        build = json.load(open(space_path)).get("space_build_seconds", 0.0)
    fit = (rounds["fit_ms"] + rounds["predict_ms"]).sum() / 1000
    evaluation = rounds["accept_ms"].sum() / 1000
    return {
        "space build": build,
        "init + validation": max(total - fit - evaluation, 0.0),
        "surrogate fit": fit,
        "candidate evaluation": evaluation,
    }


def load_eval_times(rundir):
    """Per-evaluation wall clock from pool.csv. rounds.csv reports the whole
    round, which bundles the attempts a round made before one was accepted; the
    truncation model below needs the individual evaluations."""
    pool_path = Path(rundir) / "pool.csv"
    if not pool_path.exists():
        return None
    pool = pd.read_csv(pool_path)
    if "eval_time_ms" not in pool.columns:
        return None
    ev = pool.loc[pool["eval_time_ms"].notna(), "eval_time_ms"]
    return ev.to_numpy() if len(ev) else None


def entropy_cols(batches):
    return [c for c in batches.columns if c.startswith("ent_")]


# ── Derived quantities ─────────────────────────────────────────────────────────
def amortisation_table(rounds, workers, ev=None, rng=None, draws=20000):
    """Speedup a q-wide round would reach over q sequential rounds, at equal
    evaluations spent.

    Two effects, and they pull in opposite directions:

      * the fit is paid once per round instead of once per evaluation, which is
        worth something only in proportion to fit/evaluate; and
      * the round's q evaluations overlap on `workers` workers, but the round
        cannot end before its slowest evaluation does.

    The straggler is why the parallel term is not simply min(q, workers): the
    makespan of q tasks on w workers is bounded below by both the total work
    divided across workers and the longest single task, and evaluation times
    here are heavy-tailed. Both are estimated by resampling the round's own
    observed evaluation times.
    """
    fit = (rounds["fit_ms"] + rounds["predict_ms"]).to_numpy()
    # Per-evaluation times when pool.csv supplied them: a batch holds q
    # evaluations, whereas a round's accept_ms bundles however many attempts it
    # made before one was accepted, which inflates the tail it is modelling.
    if ev is None:
        ev = rounds["accept_ms"].to_numpy()
    rng = rng or np.random.default_rng(0)
    fit_mean, ev_mean = fit.mean(), ev.mean()

    rows = []
    for q in (1, 2, 4, 8, 16, 32, 64):
        sample = rng.choice(ev, size=(draws, q), replace=True)
        # Greedy-schedule makespan bound: neither the per-worker share of the
        # total nor the longest single evaluation can be beaten.
        makespan = np.maximum(sample.sum(axis=1) / workers, sample.max(axis=1)).mean()
        rows.append(
            {
                "q": q,
                "serial_speedup": q * (fit_mean + ev_mean) / (fit_mean + q * ev_mean),
                f"parallel_speedup_w{workers}": q
                * (fit_mean + ev_mean)
                / (fit_mean + makespan),
                "batch_makespan_ms": makespan,
            }
        )
    return pd.DataFrame(rows)


def batch_speedup(ev, fit_mean, base_mean, workers, q, rng, draws=20000):
    """Speedup of one q-wide round over q sequential rounds, where evaluations
    are drawn from `ev` and the sequential baseline uses `base_mean`."""
    sample = rng.choice(ev, size=(draws, q), replace=True)
    makespan = np.maximum(sample.sum(axis=1) / workers, sample.max(axis=1)).mean()
    return q * (fit_mean + base_mean) / (fit_mean + makespan)


def truncation_sweep(ev, fit_mean, workers, fallback_ms, q, rng=None):
    """What capping the evaluation tail would buy a q-wide round.

    A batch cannot end before its slowest member, so the tail costs far more
    under batching than it does sequentially. Two ways to cut it, and the
    difference between them is not cosmetic:

      * timeout -- run until T, then fall back, paying T + fallback; and
      * predictor -- decide up front from the configuration, paying fallback.

    The timeout can never bring an evaluation below T, which puts a floor under
    the makespan; a predictor has no such floor but needs the routing decision
    to be learnable from the configuration alone.
    """
    rng = rng or np.random.default_rng(0)
    base_mean = ev.mean()

    def row(policy, thr, capped):
        total = batch_speedup(capped, fit_mean, base_mean, workers, q, rng)
        # A cheaper evaluation speeds the sequential search up too, so charging
        # all of `total` to batching would double-count. Split it: what the
        # cheaper evaluation gives on its own, and what batching adds on top.
        seq_gain = base_mean / capped.mean()
        return {
            "policy": policy,
            "thr_ms": thr,
            "truncated": float((ev > thr).mean()) if np.isfinite(thr) else 0.0,
            "eval_mean_ms": capped.mean(),
            "seq_gain": seq_gain,
            "batching_gain": total / seq_gain,
            "total_speedup": total,
        }

    rows = [row("none", np.nan, ev)]
    for p in (75, 90, 95, 99):
        thr = np.percentile(ev, p)
        for policy, cost in (
            ("timeout", thr + fallback_ms),
            ("predictor", fallback_ms),
        ):
            rows.append(row(f"{policy} p{p}", thr, np.where(ev > thr, cost, ev)))
    return pd.DataFrame(rows)


def summarise(name, rounds, batches):
    fit = rounds["fit_ms"] + rounds["predict_ms"]
    ev = rounds["accept_ms"]
    out = [f"── {name} ──", f"  rounds: {len(rounds)}"]
    out.append(
        f"  surrogate fit:  median {fit.median():8.1f} ms   "
        f"total {fit.sum() / 1000:7.1f} s   "
        f"({100 * fit.sum() / (fit.sum() + ev.sum()):4.1f}% of search)"
    )
    out.append(
        f"  evaluation:     median {ev.median():8.1f} ms   "
        f"total {ev.sum() / 1000:7.1f} s   "
        f"({100 * ev.sum() / (fit.sum() + ev.sum()):4.1f}% of search)"
    )
    out.append(f"  fit / evaluation (median): {fit.median() / ev.median():.3f}")

    sel = rounds[rounds["sel_idx"].notna()]
    if len(sel):
        out.append(
            f"  selected point's rank under mu alone:    median "
            f"{sel['rank_mu'].median():6.1f} of {sel['n_cand'].median():.0f}"
        )
        out.append(
            f"  selected point's rank under sigma alone: median "
            f"{sel['rank_sigma'].median():6.1f} of {sel['n_cand'].median():.0f}"
        )
        out.append(
            f"  selections drawn from the observed neighbourhood: "
            f"{100 * sel['from_neighbor'].mean():.0f}%"
        )

    if batches is not None and len(batches):
        for q in sorted(batches["q"].unique()):
            b = batches[batches["q"] == q]
            out.append(
                f"  q={q:3d}: effective size {b['q_eff'].mean():5.2f} "
                f"(random batch: {b['q_eff_ref'].mean():5.2f}, "
                f"ratio {(b['q_eff'] / b['q_eff_ref']).mean():4.2f})   "
                f"dispersion {b['dispersion'].mean():4.2f}   "
                f"overlap_mu {b['overlap_mu'].mean():4.2f}"
            )
    return "\n".join(out)


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_time_split(runs, ax):
    """Where each round's wall clock goes. The gap between the two series is
    what a batch can amortise; if evaluation sits above the fit, batching buys
    parallelism rather than amortisation."""
    for name, rounds, _ in runs:
        ax.plot(
            rounds["round"],
            rounds["fit_ms"] + rounds["predict_ms"],
            label=f"{name}: surrogate fit",
            lw=1.4,
        )
        ax.plot(
            rounds["round"],
            rounds["accept_ms"],
            label=f"{name}: evaluation",
            lw=1.4,
            ls="--",
        )
    ax.set_yscale("log")
    ax.set_xlabel("BO round")
    ax.set_ylabel("wall clock (ms)")
    ax.set_title("Round cost: fit vs evaluation")
    ax.legend(fontsize=7)


def plot_effective_batch(runs, ax):
    """Batch diversity relative to a uniformly drawn batch of the same size.

    Absolute effective sizes are not plotted: in a discrete space this wide,
    pairwise distances concentrate, so even a uniform batch scores far below q
    and both curves would sit flat against the axis. What carries the signal is
    how far the acquisition's batch falls below that reference -- 1.0 means it
    concentrated the batch no more than chance would.
    """
    for name, _, batches in runs:
        if batches is None or not len(batches):
            continue
        g = batches.groupby("q")
        qs = np.array(sorted(batches["q"].unique()))
        ax.plot(
            qs,
            (g["q_eff"].mean() / g["q_eff_ref"].mean()).values,
            "o-",
            label=f"{name}: effective size",
        )
        ax.plot(
            qs,
            g["dispersion"].mean().values,
            "s--",
            alpha=0.7,
            label=f"{name}: pairwise spread",
        )
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("batch size q")
    ax.set_ylabel("ratio to a uniform batch")
    ax.set_title("How much the acquisition concentrates a batch")
    ax.legend(fontsize=7)


def plot_explore_exploit(runs, ax, q=None):
    """Overlap of the acquisition's top-q with the two degenerate rankings.
    Tracking sigma means the round is uncertainty sampling; tracking mu means it
    is greedy. Where the two cross is the search's explore/exploit handover."""
    for name, _, batches in runs:
        if batches is None or not len(batches):
            continue
        qq = q if q is not None else batches["q"].max()
        b = batches[batches["q"] == qq]
        ax.plot(b["round"], b["overlap_mu"], label=f"{name}: with mu (exploit)")
        ax.plot(
            b["round"],
            b["overlap_sigma"],
            ls="--",
            label=f"{name}: with sigma (explore)",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("BO round")
    ax.set_ylabel("top-q overlap")
    ax.set_title("What the acquisition is actually ranking by")
    ax.legend(fontsize=7)


def plot_sigma_decay(runs, ax):
    """Spread of the ensemble at the selected point. A monotone collapse turns
    the acquisition into a greedy mean ranking regardless of kappa."""
    for name, rounds, _ in runs:
        sel = rounds[rounds["sel_idx"].notna()]
        ax.plot(sel["round"], sel["sel_sigma"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("BO round")
    ax.set_ylabel("ensemble sigma at selected point")
    ax.set_title("Surrogate uncertainty over the run")
    ax.legend(fontsize=7)


def plot_dim_entropy(name, batches, ax, q=None):
    """Per-dimension spread of the top-q batch over the run. Rows that fade to
    dark are dimensions the surrogate has committed to; rows that stay bright
    are still being explored."""
    cols = entropy_cols(batches)
    qq = q if q is not None else batches["q"].max()
    b = batches[batches["q"] == qq].sort_values("round")
    mat = b[cols].to_numpy().T
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
        extent=[b["round"].min(), b["round"].max(), -0.5, len(cols) - 0.5],
    )
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c[4:] for c in cols], fontsize=6)
    ax.set_xlabel("BO round")
    ax.set_title(f"{name}: per-dimension batch entropy (q={qq})")
    return im


def plot_time_budget(name, rounds, budget, outpath):
    """The RQ2 figure: where the search's wall clock goes, and the point past
    which it stops being the cost model that decides.

    Evaluation cost is flat in the number of observations -- it is one
    simulation whatever else has happened -- while the surrogate is refitted on
    everything observed so far, so its cost grows linearly. The two cross at a
    budget that bounds how far the search can usefully be pushed before the
    optimiser rather than the evaluator is what is being paid for.
    """
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    left = 0.0
    total = sum(budget.values())
    for label, secs in budget.items():
        ax0.barh([0], [secs], left=left, label=f"{label} ({100 * secs / total:.0f}%)")
        left += secs
    ax0.set_yticks([])
    ax0.set_xlabel("wall clock (s)")
    ax0.set_title(f"End-to-end search budget ({total:.0f} s)")
    ax0.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2)

    n = rounds["n_obs"].to_numpy()
    fit = (rounds["fit_ms"] + rounds["predict_ms"]).to_numpy()
    ev = rounds["accept_ms"].to_numpy()
    ax1.scatter(n, ev, s=8, alpha=0.35, color="tab:orange")
    ax1.scatter(n, fit, s=8, alpha=0.5, color="tab:blue")
    # The evaluation cost is heavy-tailed, so its level is shown as a median
    # rather than a fit; the surrogate's is a genuine linear trend.
    ev_med = np.median(ev)
    ax1.axhline(
        ev_med, color="tab:orange", lw=1.6, label=f"evaluation (median {ev_med:.0f} ms)"
    )
    slope, icpt = np.polyfit(n, fit, 1)
    cross = (ev_med - icpt) / slope if slope > 0 else np.inf
    # Carry the trend past the observed range to where it meets the evaluation
    # level, so the crossing is visible rather than merely asserted.
    extrapolated = np.isfinite(cross) and n.min() < cross < 4 * n.max()
    xmax = max(n.max(), cross * 1.12) if extrapolated else n.max()
    xs = np.linspace(n.min(), xmax, 50)
    ax1.plot(xs, slope * xs + icpt, color="tab:blue", lw=1.6, ls="--", alpha=0.6)
    ax1.plot(
        xs[xs <= n.max()],
        slope * xs[xs <= n.max()] + icpt,
        color="tab:blue",
        lw=1.6,
        label=f"surrogate fit ({slope:.2f} ms / observation)",
    )
    if extrapolated:
        ax1.axvline(cross, color="k", ls=":", lw=1.2)
        ax1.annotate(
            f"crossover\n{cross:.0f} evaluations",
            xy=(cross, ev_med),
            xytext=(-4, 10),
            textcoords="offset points",
            fontsize=7,
            ha="right",
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("observations")
    ax1.set_ylabel("per-round wall clock (ms)")
    ax1.set_title("Evaluation is flat; the surrogate is not")
    ax1.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cross


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "rundirs", nargs="+", help="directories holding rounds.csv / batchdiag.csv"
    )
    ap.add_argument(
        "-o", "--outdir", default="search_diag", help="where to write the figures"
    )
    ap.add_argument(
        "--q",
        type=int,
        default=None,
        help="batch size for the per-round panels (default: largest)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="evaluation workers assumed by the speedup estimate",
    )
    ap.add_argument(
        "--fallback-ms",
        type=float,
        default=20.0,
        help="cost of the low-fidelity evaluation in the tail sweep",
    )
    ap.add_argument(
        "--tail-q", type=int, default=64, help="batch size the tail sweep reports"
    )
    args = ap.parse_args()

    runs, evtimes = [], {}
    for d in args.rundirs:
        try:
            runs.append(load_run(d))
            evtimes[runs[-1][0]] = load_eval_times(d)
        except FileNotFoundError as e:
            print(f"skipping {d}: {e}", file=sys.stderr)
    if not runs:
        print("no runs loaded", file=sys.stderr)
        return 1

    for name, rounds, batches in runs:
        print(summarise(name, rounds, batches))
        print()
        ev = evtimes.get(name)
        print(
            f"  speedup of a q-wide round over q sequential rounds "
            f"({args.workers} workers):"
        )
        print(
            amortisation_table(rounds, args.workers, ev).to_string(
                index=False, float_format="%.2f"
            )
        )
        print()
        if ev is not None and len(ev) > 1:
            fit_mean = (rounds["fit_ms"] + rounds["predict_ms"]).mean()
            print(
                f"  capping the evaluation tail, q={args.tail_q}, "
                f"{args.workers} workers, fallback {args.fallback_ms:.0f} ms:"
            )
            print(
                truncation_sweep(
                    ev, fit_mean, args.workers, args.fallback_ms, args.tail_q
                ).to_string(index=False, float_format="%.2f")
            )
            print()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_time_split(runs, axes[0][0])
    plot_effective_batch(runs, axes[0][1])
    plot_explore_exploit(runs, axes[1][0], args.q)
    plot_sigma_decay(runs, axes[1][1])
    fig.tight_layout()
    fig.savefig(outdir / "search_diag.png", dpi=150)
    plt.close(fig)

    for (name, rounds, _), d in zip(runs, args.rundirs):
        budget = load_budget(d, rounds)
        if budget is None:
            continue
        cross = plot_time_budget(name, rounds, budget, outdir / f"rq2_time_{name}.png")
        print(f"── {name}: end-to-end budget ──")
        total = sum(budget.values())
        for label, secs in budget.items():
            print(f"  {label:24s} {secs:7.2f} s  {100 * secs / total:5.1f}%")
        print(f"  {'end to end':24s} {total:7.2f} s")
        print(f"  surrogate overtakes evaluation at ~{cross:.0f} evaluations")
        print()

    for name, _, batches in runs:
        if batches is None or not len(batches) or not entropy_cols(batches):
            continue
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = plot_dim_entropy(name, batches, ax, args.q)
        fig.colorbar(im, ax=ax, label="normalised entropy")
        fig.tight_layout()
        fig.savefig(outdir / f"dim_entropy_{name}.png", dpi=150)
        plt.close(fig)

    print(f"figures written to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
