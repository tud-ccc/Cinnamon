"""Reusable building blocks for cinm-mlir experiments: invoking cinm-opt
(exhaustive / BO / single-config eval), compiling+running on real UPMEM
hardware, parallelizing that work, and turning raw benchmark output into
net-time measurements.

Experiments (e.g. experiments/cinm1comparison/experiment.py) import this
package and define their pipeline as plain Python function calls; there is no
CLI or Makefile glue layer between the steps.
"""
