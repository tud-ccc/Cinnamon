"""Emit the paper's inline numbers as \\newcommand definitions.

Reads whatever results/*.csv exist and writes tables/numbers.tex; a source
that is not there yet contributes a comment instead of a definition, so a
partial data/ still yields a compilable include. The paper
never hard-codes one of these numbers inline: it says \\atwoRejectedCount
and friends, and this script is the single producer.

LaTeX command names contain no digits, so "A2" becomes "atwo" etc.

(Not numbers.py: a sibling numbers.py shadows the stdlib `numbers` module
for everything that runs with this directory on sys.path -- doit itself,
via tqdm.)
"""

from __future__ import annotations

import csv
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "tables" / "numbers.tex"


def _read_csv(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _cmd(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _binom_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact (Clopper-Pearson) one-sided upper confidence bound on p after
    observing k successes in n independent Bernoulli trials: the largest p
    whose chance of producing k or fewer successes is still alpha.

    At k = 0 this is the rule of three in its exact form -- 1 - alpha^(1/n),
    which 3/n approximates -- so both the headline bound and the censored one
    below come off the same estimator and their difference is only the
    successes conceded, not a change of method.

    Bisection rather than an incomplete-beta inverse: this module is stdlib
    only on purpose (it runs from doit, ahead of the plotting environment),
    and the binomial CDF is a sum of k + 1 terms here.
    """
    if k >= n:
        return 1.0

    def cdf(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _hyper_upper(k: int, n: int, m: int, alpha: float = 0.05) -> int:
    """Exact one-sided upper confidence bound on K -- the number of configs in
    a feasible space of m that beat the reference -- after drawing n of them
    without replacement and finding k that do: the largest K whose chance of
    producing k or fewer successes is still alpha. Returns a count, not a
    fraction, since the caller needs both K and K/m.

    The draw is a lazy Fisher-Yates over the enumerated feasible set, so the
    trials are exchangeable but not independent and the count is
    hypergeometric. That buys a ceiling the binomial does not have:
    P(X <= k | K) is exactly zero once K > m - n + k, because the space
    cannot hide more winners than the draw left untouched. The binomial keeps
    assigning probability to "missed them all" past that point, which is why
    it is the looser of the two and by how much: the gap is governed by the
    sampling fraction n/m, negligible below a few percent and worth a factor
    of two by the time the draw covers three quarters of the space.

    Bisection over K, which P(X <= k | K) is non-increasing in -- the same
    shape as _binom_upper, and stdlib for the same reason. Exact integer
    binomials rather than lgamma: the terms are a few thousand digits at
    m = 1.5M and the bound lands in a paper.
    """
    if n >= m:
        # The sample is the space. Exactly k configs beat the reference;
        # there is nothing left for a confidence bound to cover.
        return k

    total = math.comb(m, n)

    def cdf(n_win: int) -> float:
        return (
            sum(
                math.comb(n_win, i) * math.comb(m - n_win, n - i)
                for i in range(k + 1)
                if i <= n_win and n - i <= m - n_win
            )
            / total
        )

    lo, hi = k, m
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def a2_numbers() -> list[str]:
    """A2's two fractions: rejected-but-lowers (the constraint system's
    false negatives) and accepted-but-fails (must be zero). Aggregated over
    every probed function; the per-function table stays in a2.csv."""
    rows = _read_csv("a2.csv")
    if rows is None:
        return ["% a2.csv not assembled yet -- run `doit a2 assemble_a2`"]
    rejected = sum(
        int(r["n_rejected_lowers"]) + int(r["n_rejected_fails"]) for r in rows
    )
    lowers = sum(int(r["n_rejected_lowers"]) for r in rows)
    accepted_fails = sum(int(r["n_accepted_fails"]) for r in rows)
    out = [
        _cmd("atwoRejectedCount", rejected),
        _cmd("atwoRejectedLowersCount", lowers),
        _cmd("atwoAcceptedButFails", accepted_fails),
    ]
    if rejected:
        out.append(_cmd("atwoRejectedLowersPct", f"{100.0 * lowers / rejected:.1f}"))
        # Rule of three: when nothing in the rejected region lowers, the 95%
        # upper bound on the false-negative rate is 1 - alpha^(1/n), which
        # 3/n approximates -- pooled over every probed function, so it is
        # small (0.038% at 7800) and needs significant digits, not decimal
        # places. Binomial rather than the hypergeometric E1 uses: a2 probes
        # each function's rejected region and a2.csv does not record how
        # large that region is, so there is no population size to correct
        # against and the with-replacement reading is the conservative one.
        if lowers == 0:
            out.append(
                _cmd(
                    "atwoFalseNegativeBoundPct",
                    f"{100.0 * _binom_upper(0, rejected):.2g}",
                )
            )
    return out


TRANSCRIBED_SYSTEMS = ("atim_published_transcribed", "atim_reproduced_transcribed")


def sample_census_numbers() -> list[str]:
    """The shared uniform sample's census, and the percentile bound that
    survives the simulation timeout.

    The timeout is a budget on the *simulator*, and the percentile is computed
    over *measured* runtime, so a timed-out configuration is not missing from
    the sample: it keeps its slot and B2 still benchmarks it. What it loses is
    its predicted cost. The bound below therefore comes in two readings --
    the one the measured rows support, and the one that concedes every
    timed-out row as beating the reference, which is the strongest claim that
    needs no argument about what the simulator was doing when it ran out.

    Bounds are per benchmark, and the paper quotes the weakest, since each
    benchmark is a different space and "the top X%" has to hold for all of
    them. That makes the headline number the one from the largest space; the
    tightest is emitted alongside it, because the spaces span three orders of
    magnitude and a single figure hides that the small ones are pinned far
    harder. Both are hypergeometric, so the space sizes are emitted too --
    without m the bound is not a number a reviewer can recompute.

    Failed draws are reported separately and must be zero: unlike a timeout,
    a failure leaves no row, so the sample would be a draw from the space
    minus a region nobody characterised. Exhausted spaces are counted for the
    same reason and in the same spirit -- if a draw ever covers its whole
    space the claim there is enumeration, not inference, and the confidence
    language has to come off that benchmark rather than be quoted at zero."""
    census = _read_csv("sample_census.csv")
    if census is None:
        return ["% sample_census.csv not assembled yet -- run `doit assemble`"]

    n_timed_out = sum(int(r["n_timed_out"]) for r in census)
    n_failed = sum(int(r["n_failed"]) + int(r["n_over_budget"]) for r in census)
    out = [
        _cmd("eoneSampleN", census[0]["n_requested"]),
        _cmd("eoneSampleTimeoutCount", n_timed_out),
        _cmd("eoneSampleDroppedCount", n_failed),
    ]

    rows = _read_csv("e1.csv")
    if rows is None:
        out.append("% e1.csv not assembled yet -- percentile bounds pending")
        return out

    # Per (benchmark, fn): how many sampled configurations measured faster
    # than the best transcribed ATiM point, out of how many were drawn, and
    # out of how many the feasible space holds.
    by_fn = {(r["benchmark"], r["fn_name"]): r for r in census}
    plain, censored, sizes, n_exhausted = [], [], [], 0
    for (bench, fn), crow in sorted(by_fn.items()):
        sub = [r for r in rows if r["benchmark"] == bench and r["fn_name"] == fn]
        sample = [float(r["total_ms"]) for r in sub if r["system"] == "sample"]
        ref = [float(r["total_ms"]) for r in sub if r["system"] in TRANSCRIBED_SYSTEMS]
        if not sample or not ref:
            continue
        n = len(sample)
        m = int(crow["space_size"])
        n_to = int(crow["n_timed_out"])
        k = sum(1 for t in sample if t < min(ref))
        if n >= m:
            n_exhausted += 1
        sizes.append(m)
        plain.append(_hyper_upper(k, n, m) / m)
        censored.append(_hyper_upper(min(k + n_to, n), n, m) / m)

    if not plain:
        out.append("% e1.csv has no sample/transcribed pair yet -- bounds pending")
        return out
    out += [
        _cmd("eoneSamplePercentileBoundPct", f"{100.0 * max(plain):.2g}"),
        _cmd("eoneSamplePercentileBoundBestPct", f"{100.0 * min(plain):.2g}"),
        _cmd("eoneSamplePercentileBoundCensoredPct", f"{100.0 * max(censored):.2g}"),
        _cmd("eoneSampleSpaceSizeMin", f"{min(sizes):,}"),
        _cmd("eoneSampleSpaceSizeMax", f"{max(sizes):,}"),
        _cmd("eoneSampleExhaustedCount", n_exhausted),
    ]
    return out


def e1_numbers() -> list[str]:
    rows = _read_csv("e1.csv")
    if rows is None:
        return [
            "% e1.csv not assembled yet (Phase 1) -- best-vs-median and percentile pending"
        ]
    # Filled in when Phase 1's assembler defines e1.csv's columns.
    return ["% e1.csv present but its numbers are not wired yet"]


def rq2_numbers() -> list[str]:
    rows = _read_csv("rq2.csv")
    if rows is None:
        return ["% rq2.csv not assembled yet (Phase 2) -- A3 seed spread pending"]
    return ["% rq2.csv present but its numbers are not wired yet"]


def main() -> None:
    lines = [
        "% Generated by evaluation/paper_numbers.py -- do not edit. One",
        "% \\newcommand per number the paper quotes inline; missing inputs",
        "% degrade to comments so this file always compiles.",
        *a2_numbers(),
        *sample_census_numbers(),
        *e1_numbers(),
        *rq2_numbers(),
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
