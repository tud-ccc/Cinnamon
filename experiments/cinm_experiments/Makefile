UPMEM_HOME ?= $(error UPMEM_HOME is not set)

# The host lowering below is a pipeline of processes. Without pipefail only
# the last one's exit status is seen: a cinm-opt that dies mid-module feeds
# llc a truncated module, llc succeeds, and make reports success over a stub
# object file. DELETE_ON_ERROR then removes that stub -- left behind it is
# newer than its source, so the next make would take it as up to date.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DELETE_ON_ERROR:

ROOT         = ../..
CINM_OPT     = $(ROOT)/build/bin/cinm-opt
CINM_TRANS   = $(ROOT)/build/bin/cinm-translate
CLANG_BASE   = $(ROOT)/third-party/llvm/build/bin
COMPILE_DPU  = $(ROOT)/testbench/lib/compile_dpu.sh

# Primitive name (red, gemv, …). Selects the bench driver under experiments/bench/.
PRIM         ?= red
BENCH_DRIVER ?= $(ROOT)/experiments/bench/$(PRIM).cpp

# Single-function source module, already lowered to the upmem dialect by the
# caller (e.g. cinmopt.py / cinm1.py). This makefile takes over from there:
# host lowering + DPU kernel compilation + linking the bench binary.
SRC_MLIR ?= $(error SRC_MLIR is not set)
BENCH_FN ?= $(error BENCH_FN is not set)

IR_DIR  ?= ir
BIN_DIR ?= bin

CXXFLAGS = -O3 -std=c++17 \
           -I$(ROOT)/runtime/Upmem \
           -I$(UPMEM_HOME)/include/dpu

# openblas backs the bench drivers' golden references (experiments/bench/
# common.hpp): a run checks its output against an independent implementation
# before its timings are written, unless BENCH_CHECK=0 says otherwise.
LDFLAGS = -lUpmemDialectRuntime -ldpu -ldpuverbose -lopenblas -Wl,-s \
          -L$(UPMEM_HOME)/lib \
          -L$(ROOT)/build/lib

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
    --convert-linalg-to-affine-loops \
		--affine-scalrep --affine-super-vectorize=vectorize-reductions\ virtual-vector-size=8 --affine-expand-index-ops-as-affine \
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
	cinm-opt $< --upmem-annotate-costs="simulator=hybrid eval-timeout-ms=400 costs-csv=$@"

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
$(IR_DIR)/%.o: $(IR_DIR)/%.mlir
	$(CINM_OPT) $< $(LOWER_PIPELINE) $(TO_LLVM_PIPELINE) -o - \
	    | $(CINM_TRANS) --allow-unregistered-dialect --mlir-to-llvmir -o - \
	    | $(CLANG_BASE)/opt -O3 | $(CLANG_BASE)/llc -O3 -filetype=obj -relocation-model=pic -o $@

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
	clang++ $(CXXFLAGS) -DBENCH_FN=$(BENCH_FN) $(BENCH_DRIVER) \
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
