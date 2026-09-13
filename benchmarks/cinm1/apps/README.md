# cinm1 drivers

These share `benchmarks/common.hpp` with the prim and multiop suites -- the
operand distribution, the golden references, and the verification are the same
code -- but they do not go through its `bench::run<Op>()`.

That template is one kernel per process by construction: `BENCH_FN` is a
compile-time name, and the experiment pipelines build a binary per
configuration. This suite is built the other way round, one binary per source
module, because a cinm1 module holds every DIMM-count variant of its kernel and
the makefile compiles the module once. So a driver here loops over its module's
kernels itself and shares `bench::time_mean_ms` for the timing.

Each driver therefore states, per kernel, the shape to allocate and the
reference to check against -- the shapes must match the `func.func` signatures
in the suite's `.mlir`, since nothing derives one from the other.
