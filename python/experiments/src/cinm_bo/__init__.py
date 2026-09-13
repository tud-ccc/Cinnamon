"""Command-line tools for running and diagnosing the accelerator search.

`python -m cinm_bo <subcommand>`: drive cinm-opt's Bayesian search over a
benchmark (run, seeds, exhaustive), then look at what it did (plot, view,
analyze). The diagnostics are the only readers of the search's rounds.csv,
batchdiag.csv, training.csv and validation.csv dumps, which is why they live
with the compiler rather than with any one experiment.

The analysis library they share is cinm_experiments.
"""
