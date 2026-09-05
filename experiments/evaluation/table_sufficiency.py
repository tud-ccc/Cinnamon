"""tab:sufficiency (e1.csv): does the space contain ATiM-grade points, and
does the search find them?

Per (benchmark, fn): (a) ATiM offline, (a') its transcription measured
under our codegen (best reading when several -- the paper measures every
reading of an ambiguous transcription), (b) space best -- the best
MEASURED point of any arm, sample, top-k or a search pick, since an
exhibited witness does not care where it came from -- then the sample
median, and the search reported as its MEDIAN seed with the spread across
seeds and the regret (c)/(b). Rows run by benchmark, then by operand size
within each; a hole prints as --.

The column adjacencies are the argument (the paper's E1 block spells it
out): (b) reads against (a') -- sufficiency, against the external
baseline. The sample median sits next to (b) as the best-vs-median spread
-- C1's "the space is treacherous", never a bar the search cleared. The
search reads against the space through regret alone; (b) contains the
search picks, so a (b)-vs-(c) time comparison would be a max read against
a median and is not printed. No percentile column: search-best saturates
at 0.0% the moment the search works; the (a') percentile with its CI is
paper_numbers.py's job, in prose. Median seed, never best-of-N: spreads
reached ~8x in the first campaign, and the spread column is what keeps
that honest.

Cells whose sample is still filling are wrapped in \\textcolor{red} -- the
paper must load xcolor. See _red for why partial is worse than noisy here.

(a) and (a') are a pair per ATiM variant: the schedules published with
their artifact, and the ones their autotuner found on this machine. Each
gap is to a specific configuration, and taking the better of the two would
decompose the gap to a configuration neither run produced.

Point cells -- the four ATiM columns and space best -- are `net (total)`
in ms; the caption has to say so. Net is the amortized number the rest of
the pipeline is stated in; total adds back the operand movement that row's
system performs but does not report (`excluded_transfer_ms`: ATiM's
`pragma_explicit_h2d` operands, our `cinm.static` ones). The pair is
there because the two systems do not draw that line in the same place --
ATiM excludes `va`'s and `geva`'s inputs, where we have no weight to
amortize against and exclude nothing -- so a comparison of net against
net is sound for some rows and not others, and which is which should be
visible in the table rather than argued in prose. Where the two
conventions agree the pair prints the same number twice, which is itself
the thing worth seeing. The statistic cells (sample median, search
median) are net only: a median is not a row, so it has no single
excluded-transfer number to add back.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from _reporting import load_or_skip, parse_dirs, tex, write_tex


_SIZE_UNITS = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}


def _fn_sort_key(fn: str) -> tuple[str, int]:
    """(stem, operand bytes), so a function's rows run 4MB, 64MB, 256MB, 512MB.

    These names end in a size, and sorting the string puts 256MB above 4MB --
    which reads as a shuffle in a column that is otherwise a size sweep, and
    hides the trend the sweep exists to show. A name carrying no size keeps
    its text order among its peers rather than being forced to one end."""
    m = re.fullmatch(r"(.*?)_(\d+)(B|KB|MB|GB)", fn)
    return (m.group(1), int(m.group(2)) * _SIZE_UNITS[m.group(3)]) if m else (fn, -1)


def _best(sub) -> tuple[float, float | None] | None:
    """The best row of `sub`, as (net, un-amortized total).

    Chosen by net, because that is what the pipeline optimizes and what the
    percentile is computed in. The total is then read off that same row
    rather than minimized in its own right, which would be free to answer
    with a different configuration than the one the cell names.

    A row whose `excluded_transfer_ms` is absent or not finite has no total
    to state -- not a total equal to its net. Zero is a different answer and
    an informative one: it is what a function with nothing to amortize
    (`va`, `geva`) records.
    """
    if not len(sub):
        return None
    row = sub.loc[sub["total_ms"].idxmin()]
    net = float(row["total_ms"])
    excluded = row.get("excluded_transfer_ms")
    if excluded is None or not np.isfinite(excluded):
        return net, None
    return net, net + float(excluded)


def _fmt(v: tuple[float, float | None] | None) -> str:
    """`net (total)`, each half printing as -- when it is a hole."""
    if v is None:
        return "--"
    net, total = v
    return f"{net:.2f} ({total:.2f})" if total is not None else f"{net:.2f} (--)"


def _pool_sizes(results_dir) -> dict[tuple[str, str], int]:
    """(benchmark, fn) -> the shared draw's size, from sample_census.csv.

    The second results/ file this script reads, and the only one it can do
    without: absent, every sample-derived cell is marked unverified rather
    than the table refusing to render. `n_rows` is the census's own reading
    of pool.csv, which is what B2 measures, so it is the count a finished
    sample stack has to reach."""
    path = results_dir / "sample_census.csv"
    if not path.exists():
        print("sample_census.csv not assembled -- sample completeness unknown")
        return {}
    census = pd.read_csv(path)
    return {
        (row.benchmark, row.fn_name): int(row.n_rows) for row in census.itertuples()
    }


def _red(cell: str, verified: bool) -> str:
    """Mark a cell whose sample is still filling. Needs xcolor in the paper.

    A partial stack is not merely a weaker version of a finished one. The
    pool is emitted in `dpus` order, so the rows measured first are the
    low-DPU corner of the space -- the worst configurations by construction.
    That skews every sample-derived cell, each in its own direction: space
    best is weak (nothing good measured yet), which flatters regret; the
    sample median is drawn from the worst corner, which slanders the space;
    and neither is a random subsample of the draw the (a') percentile
    machinery assumes. Red says provisional, not merely noisy.

    A hole is left alone: -- already says the stack has nothing to show, and
    colouring it would spend the reader's attention on the rows that make no
    claim at all."""
    if verified or cell == "--":
        return cell
    return f"\\textcolor{{red}}{{{cell}}}"


def main() -> None:
    results_dir, out_dir = parse_dirs("tables")
    df = load_or_skip(results_dir, "e1.csv")
    if df is None:
        return

    # (offline system, transcribed system) per ATiM variant, in column order.
    VARIANTS = [
        ("atim_published", "atim_published_transcribed"),
        ("atim_reproduced", "atim_reproduced_transcribed"),
    ]
    lines = [
        "% tab:sufficiency, generated by table_sufficiency.py -- do not edit",
        "\\begin{tabular}{ll" + "rr" * len(VARIANTS) + "rrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{ATiM published}"
        " & \\multicolumn{2}{c}{ATiM reproduced} & & & & & \\\\",
        "benchmark & size (MB) & measured & transcribed & measured & transcribed"
        " & space best & sample med & search med & spread & regret \\\\",
        "\\midrule",
    ]
    pool_sizes = _pool_sizes(results_dir)
    groups = sorted(
        df.groupby(["benchmark", "fn_name"]),
        key=lambda kv: (kv[0][0], _fn_sort_key(kv[0][1])),
    )
    for (bench, fn), sub in groups:
        by = {s: g for s, g in sub.groupby("system")}
        sample = by.get("sample")
        search = by.get("search")
        # The exhibited witness: best measured point of any arm, the search's
        # own picks included. A witness does not care where it came from, and
        # a "best known" that excludes the search reads false the moment a
        # pick wins its row. This is also why no cell compares the search
        # against space best: (b) contains (c), so regret is the comparison,
        # made once, in its own column.
        witness = sub[sub["system"].isin(["sample", "topk", "search"])]
        space_best = _best(witness)
        n_sample = len(sample) if sample is not None else 0
        n_pool = pool_sizes.get((bench, fn))
        verified = n_pool is not None and n_sample >= n_pool
        sample_med = f"{sample['total_ms'].median():.2f}" if n_sample else "--"
        search_med, spread, regret = "--", "--", "--"
        if search is not None and len(search):
            # One row per seed (identical picks are not deduplicated), so
            # the median over rows is the median seed -- the reported
            # configuration per A3's rule, never best-of-N. The spread is
            # what keeps that rule honest in print.
            med = float(search["total_ms"].median())
            search_med = f"{med:.2f}"
            spread = (
                f"{search['total_ms'].max() / search['total_ms'].min():.1f}$\\times$"
            )
            if space_best is not None:
                regret = _red(f"{med / space_best[0]:.2f}$\\times$", verified)
        bench_clean = bench.removeprefix("prim_")
        fn_clean = fn.removeprefix(bench_clean + "_").removesuffix("MB")
        lines.append(
            " & ".join(
                [
                    tex(bench_clean),
                    tex(fn_clean),
                    *(
                        _fmt(_best(by.get(system, sub.iloc[0:0])))
                        for variant in VARIANTS
                        for system in variant
                    ),
                    _red(_fmt(space_best), verified),
                    _red(sample_med, verified),
                    search_med,
                    spread,
                    regret,
                ]
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(out_dir, "sufficiency.tex", lines)


if __name__ == "__main__":
    sys.exit(main())
