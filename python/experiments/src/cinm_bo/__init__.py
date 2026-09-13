"""Command-line tools for running and diagnosing the accelerator search.

`python -m cinm_bo <command>`: produce a dump directory (space, sample,
search, exhaustive), then read one back (plot, diag, analyze, view). The
diagnostics are the only readers of the search's rounds.csv, batchdiag.csv,
training.csv and validation.csv, which is why they live with the compiler
rather than with any one experiment.

The analysis library they share is cinm_experiments.
"""
