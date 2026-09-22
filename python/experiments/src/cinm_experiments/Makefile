UPMEM_HOME ?= $(error UPMEM_HOME is not set)

# The host lowering below is a pipeline of processes. Without pipefail only
# the last one's exit status is seen: a cinm-opt that dies mid-module feeds
# llc a truncated module, llc succeeds, and make reports success over a stub
# object file. DELETE_ON_ERROR then removes that stub -- left behind it is
# newer than its source, so the next make would take it as up to date.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DELETE_ON_ERROR:

# Where an installed Cinnamon put itself. compile_run.py passes all three,
# from cinnamon.paths, and they are not derived from one another: this file
# ships inside a package whose depth below anything differs between an
# editable install and a wheel, so counting ".." here would be right in one
# case and silently wrong in the other. Set them by hand to run make from this
# directory -- `python -c 'from cinnamon import paths; print(paths.bin_dir())'`
# says what the environment holds.
#
# opt and llc live in CINM_BIN too, from whichever LLVM this was built
# against, because IR emitted by cinm-translate is not necessarily readable by
# another LLVM's tools. The host driver's compiler is a separate question --
# it compiles C++, it does not read our IR -- and is settled at BENCH_CXX.
CINM_BIN    ?= $(error CINM_BIN is not set)
BENCH_DIR   ?= $(error BENCH_DIR is not set)
RT_INCLUDE  ?= $(CINM_BIN)/../include/cinm-mlir/runtime/Upmem
RT_LIB_DIR  ?= $(CINM_BIN)/../lib
# Each cinm-opt call below applies a few passes, so cinm-opt's half-second
# startup dominates it. cinm-opt-client runs the call in a cinm-opt fork
# server that has already paid it, and runs cinm-opt directly whenever there
# is no server to use. An install older than the client has only cinm-opt.
CINM_OPT     = $(or $(wildcard $(CINM_BIN)/cinm-opt-client),$(CINM_BIN)/cinm-opt)
CINM_TRANS   = $(CINM_BIN)/cinm-translate
COMPILE_DPU  = $(CINM_BIN)/cinm-compile-dpu

# Benchmark name: a primitive (red, gemv, …) or a multi-op workload
# (2mm_seq, …). Selects the driver, which lives beside its source module in
# whichever family directory holds it -- so the search rather than a fixed
# path, and an error rather than an empty BENCH_DRIVER and a baffling link.
PRIM         ?= red
# Operator suites keep one driver per file under a suite directory; the
# end-to-end models keep everything about a model -- source, schedule,
# makefile, driver -- in a directory of its own under models/.
# The suites are named rather than globbed: cinm1/ is the original CINM 1.0
# suite with its own Makefile and drivers of its own conventions (cinm1/va.cpp
# calls va_8 and va_16), and a glob over every suite directory picked it ahead
# of prim/va.cpp -- while the CINM 1.0 *arm* of the comparison lowers the PrIM
# functions and is benched with the PrIM drivers like everything else here.
BENCH_SUITES ?= prim multiop
BENCH_DRIVER ?= $(firstword $(wildcard $(foreach s,$(BENCH_SUITES),$(BENCH_DIR)/$(s)/$(PRIM).cpp) $(BENCH_DIR)/models/*/$(PRIM).cpp))
ifeq ($(BENCH_DRIVER),)
  $(error No bench driver for PRIM=$(PRIM) under $(BENCH_DIR)/{$(BENCH_SUITES)}/ or $(BENCH_DIR)/models/*/)
endif

# Single-function source module, already lowered to the upmem dialect by the
# caller (e.g. cinmopt.py / cinm1.py). This makefile takes over from there:
# host lowering + DPU kernel compilation + linking the bench binary.
SRC_MLIR ?= $(error SRC_MLIR is not set)
BENCH_FN ?= $(error BENCH_FN is not set)

IR_DIR  ?= ir
BIN_DIR ?= bin

# The host driver's compiler: the system's, named explicitly.
#
# Not a bare `clang++`, because UPMEM_HOME/bin is on PATH whenever this runs
# and the SDK ships a clang 12 under that name -- a bare name hands the
# binaries whose timings are the results to whatever PATH happens to put
# first (common.sh refuses a bare clang for the same reason).
#
# And deliberately not $(CXX), which inside the pixi environment is conda's
# clang. That one cannot link this binary at all: it drags in its own sysroot,
# while the SDK's libdpu.so resolves against the system's libelf and glibc, so
# the link fails on elf_begin@ELFUTILS_1.0 and dlopen@GLIBC_2.34. The host
# driver has to be built with the toolchain the SDK was.
BENCH_CXX   ?= c++
ifneq ($(UPMEM_HOME),)
  ifneq ($(findstring $(UPMEM_HOME),$(BENCH_CXX)),)
    $(error BENCH_CXX is $(BENCH_CXX), inside the UPMEM SDK -- that is its \
            clang 12, not a host compiler. Set BENCH_CXX.)
  endif
endif

# openblas and numa come from the environment when there is one -- they are
# declared in pixi.toml -- and from the system otherwise. Without the explicit
# -I/-L the system copies win even inside the environment, since nothing else
# puts these on the search path. These are plain C libraries loaded alongside
# the SDK's, not a toolchain, so taking them from the environment is fine
# where taking the compiler from it is not (see BENCH_CXX).
ifneq ($(CONDA_PREFIX),)
  CONDA_CXXFLAGS = -isystem $(CONDA_PREFIX)/include
  CONDA_LDFLAGS  = -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

CXXFLAGS = -O3 -std=c++17 \
           -I$(RT_INCLUDE) \
           -I$(UPMEM_HOME)/include/dpu \
           $(CONDA_CXXFLAGS)

# The processor the host object is generated for. Empty means llvm's
# `generic`, the 2003 x86-64 baseline: SSE2, 16-byte vectors, which is what
# the objects held whatever --affine-super-vectorize produced above them.
# The bench machine is a Xeon Silver 4216, which LLVM calls `cascadelake`
# and which brings AVX-512 and VNNI (the instruction the int8 benchmarks
# want). Named rather than hardcoded because the binary then requires that
# processor.
HOST_MCPU ?= cascadelake
MCPU_FLAG = $(if $(HOST_MCPU),-mcpu=$(HOST_MCPU),)

# Elements per vector for --affine-super-vectorize. Elements, not bytes, so
# the width it asks for depends on the element type: 16 fills a zmm with
# i32, while an i8 program wants 64 and gets an eighth of one at this
# setting. The pass vectorises the contiguous loop, not the reduction it
# accumulates into, so this has to divide the innermost result dimension --
# every benchmark here has a power of two there -- and the reduction's own
# trip count is free. HOST_VECTORIZE=0 drops the pass and leaves
# vectorisation to llvm's loop vectoriser, which on this IR emits nothing
# wider than xmm even with -mcpu.
HOST_VECTOR_SIZE ?= 16
HOST_VECTORIZE ?= 1
SUPER_VECTORIZE = $(if $(filter-out 0,$(HOST_VECTORIZE)),--affine-super-vectorize=vectorize-reductions\ virtual-vector-size=$(HOST_VECTOR_SIZE),)

# openblas backs the bench drivers' golden references (benchmarks/common.hpp):
# a run checks its output against an independent implementation before its
# timings are written, unless BENCH_CHECK=0 says otherwise.
# numa backs bench::interleaved_pages, which spreads the pages of every
# buffer a transfer touches across the nodes, so the measured bandwidth is
# the transfer's and not an artifact of single-node page placement (see the
# comment on interleaved_pages).
LDFLAGS = -lUpmemDialectRuntime -ldpu -ldpuverbose -lopenblas -lnuma -Wl,-s \
          -L$(UPMEM_HOME)/lib \
          -L$(RT_LIB_DIR) \
          $(CONDA_LDFLAGS)

# Passes to lower a upmem-level module to host LLVM IR.
# Bufferize function boundaries first, then the standard lowering sequence.
# buffer-results-to-out-params needs modify-public-functions: the benchmark
# entry point is public, and without it the pass leaves the function returning
# a freshly allocated memref. The bench drivers pass an out buffer and read the
# result back from it, so the returned buffer would be dropped on the floor and
# the output never written.
# Unwrapping puts the per-block program loads onto the actual DPU-set values;
# dedup then gives structurally identical programs one symbol, and the hoist
# pass collapses each set's uniform loads into a single one at its
# allocation -- a resident program is loaded once per set lifetime, not once
# per member block. (The DPU-side %.kernels.mlir target compiles from the
# pre-LOWER module, so deduplicated symbols still resolve to a binary.)
LOWER_PIPELINE = \
    --cinm-unwrap-compute-blocks \
    --upmem-dedup-kernels \
    --upmem-hoist-load-programs \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    --buffer-results-to-out-params="hoist-static-allocs modify-public-functions" \
    --cnm-promote-host-allocs \
    --convert-linalg-to-affine-loops \
		--affine-scalrep $(SUPER_VECTORIZE) --affine-expand-index-ops-as-affine \
		--convert-vector-to-amx --convert-vector-to-scf --convert-vector-to-llvm \
    --canonicalize --cse
#     --cinm-unwrap-compute-blocks

TO_LLVM_PIPELINE = \
    --convert-upmem-to-llvm \
    --convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
    --expand-strided-metadata --memref-expand --lower-affine \
    --convert-scf-to-cf \
    --convert-to-llvm \
    --cse --reconcile-unrealized-casts --canonicalize \
    --llvm-legalize-for-export --canonicalize --cse

.PHONY: bench-single costs-only clean relink

.PRECIOUS: $(IR_DIR)/%.mlir $(IR_DIR)/%.kernels.mlir $(IR_DIR)/%.dpu.c

$(IR_DIR) $(BIN_DIR):
	mkdir -p $@

$(IR_DIR)/cost.csv: $(SRC_MLIR) | $(IR_DIR)
	$(CINM_OPT) $< --upmem-annotate-costs="simulator=fast eval-timeout-ms=1000 costs-csv=$@"

# Predicted costs only -- no DPU kernel compilation, no host object/link step.
# Just the cinm-opt cost-annotation pass above, so this is much cheaper than
# bench-single when only the cost breakdown is needed (e.g. validating/
# retraining the cost model against a much larger config sweep than what
# actually gets compiled to a real binary).
costs-only: $(IR_DIR)/cost.csv

$(IR_DIR)/$(BENCH_FN).mlir: $(SRC_MLIR) | $(IR_DIR)
	cp $< $@

# ─── Host compilation ────────────────────────────────────────────────────────

# Source module straight to object file, in one process each.
#
# A cinm-opt costs ~0.5s before it looks at any IR (`cinm-opt --version`, which
# does nothing else, takes that long) while each of these two pipelines takes
# under 0.1s to actually run, so running them as one invocation is nearly a
# free 0.5s. cinm-translate is the same story at ~0.4s. Piping rather than
# writing %.final.mlir/%.llvm.mlir/%.ll is worth little by itself, but nothing
# reads them: the DPU side branches off %.mlir, not off these.
#
# Byte-identical to what the four-target chain produced. Keep it that way --
# the object file ends up in a benchmark whose timings are the results.
#
# The medium code model, because the repack staging globals of a model with
# resident weights add up to more than the 2 GB the small model can address
# (a 3B decode step stages 3.4 GB of them): data above the large-data
# threshold moves to .ldata behind 64-bit addressing, which costs one address
# materialisation per global outside the loops that use it, and everything
# else -- the runtime's own globals included -- stays where it was.
$(IR_DIR)/%.o: $(IR_DIR)/%.mlir
	$(CINM_OPT) $< $(LOWER_PIPELINE) $(TO_LLVM_PIPELINE) -o - \
	    | $(CINM_TRANS) --allow-unregistered-dialect --mlir-to-llvmir -o - \
	    | $(CINM_BIN)/opt -O3 $(MCPU_FLAG) | $(CINM_BIN)/llc -O3 $(MCPU_FLAG) -filetype=obj -relocation-model=pic -code-model=medium -o $@

# The stages the rule above pipes through, for when a compile fails and the
# intermediate IR is what you want to look at. Nothing depends on these; ask
# for one by name (`make ir/foo.final.mlir`).
$(IR_DIR)/%.final.mlir: $(IR_DIR)/%.mlir
	$(CINM_OPT) $< $(LOWER_PIPELINE) -o $@

$(IR_DIR)/%.llvm.mlir: $(IR_DIR)/%.final.mlir
	$(CINM_OPT) $< $(TO_LLVM_PIPELINE) -o $@

$(IR_DIR)/%.ll: $(IR_DIR)/%.llvm.mlir
	$(CINM_TRANS) --allow-unregistered-dialect --mlir-to-llvmir $< -o $@

# ─── DPU kernel compilation ──────────────────────────────────────────────────

$(IR_DIR)/%.kernels.mlir: $(IR_DIR)/%.mlir
	$(CINM_OPT) --upmem-legalize-for-translation --lower-affine --cse --canonicalize $< -o $@

$(IR_DIR)/%.dpu.c: $(IR_DIR)/%.kernels.mlir
	$(CINM_TRANS) --allow-unregistered-dialect --mlir-to-upmem-cpp $< -o $@

$(BIN_DIR)/$(BENCH_FN): $(IR_DIR)/$(BENCH_FN).dpu.c | $(BIN_DIR)
	$(COMPILE_DPU) $< $@

# ─── Final host binary ───────────────────────────────────────────────────────

define LINK_BENCH
	$(BENCH_CXX) $(CXXFLAGS) -DBENCH_FN=$(BENCH_FN) $(BENCH_DRIVER) \
	    $(IR_DIR)/$(BENCH_FN).o $(LDFLAGS) \
	    -o $(BIN_DIR)/bench_$(BENCH_FN)
endef

bench-single: $(IR_DIR)/$(BENCH_FN).o $(BIN_DIR)/$(BENCH_FN) $(IR_DIR)/cost.csv | $(BIN_DIR)
	$(LINK_BENCH)

# Just the final link step (e.g. after editing BENCH_DRIVER or the runtime
# lib) -- skips bench-single's .o/DPU-kernel/cost.csv prerequisites, and
# (being prerequisite-less and phony) always relinks instead of relying on
# make's timestamp check against the already-built $(IR_DIR)/$(BENCH_FN).o.
relink: | $(BIN_DIR)
	$(LINK_BENCH)

clean:
	rm -rf $(IR_DIR) $(BIN_DIR)
