UPMEM_HOME ?= $(error UPMEM_HOME is not set)

ROOT         = ../../..
CINM_OPT     = $(ROOT)/build/bin/cinm-opt
CINM_TRANS   = $(ROOT)/build/bin/cinm-translate
CLANG_BASE   = $(ROOT)/third-party/llvm/build/bin
DPU_CC       = $(UPMEM_HOME)/bin/dpu-upmem-dpurte-clang
COMPILE_DPU  = $(ROOT)/testbench/lib/compile_dpu.sh

SRC_MLIR = $(ROOT)/experiments/data/prim_red/out.mlir
FUNCS    = red_4MB red_64MB red_256MB red_512MB

IR_DIR  = ir
BIN_DIR = bin
OUT_DIR = output

HOST_OBJS = $(FUNCS:%=$(IR_DIR)/%.o)

CXXFLAGS = -O3 -std=c++17 \
           -I$(ROOT)/runtime/Upmem \
           -I$(UPMEM_HOME)/include/dpu

LDFLAGS = -lUpmemDialectRuntime -ldpu -ldpuverbose -Wl,-s \
          -L$(UPMEM_HOME)/lib \
          -L$(ROOT)/build/lib

# Passes to lower a upmem-level module to host LLVM IR.
# Bufferize function boundaries first, then the standard lowering sequence.
LOWER_PIPELINE = \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
		--buffer-results-to-out-params=hoist-static-allocs \
    --canonicalize --cse \
    --cinm-unwrap-compute-blocks \
    --convert-upmem-to-llvm \
    --convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
    --expand-strided-metadata --memref-expand --lower-affine \
    --convert-scf-to-cf \
    --convert-to-llvm \
    --cse --reconcile-unrealized-casts --canonicalize \
    --llvm-legalize-for-export --canonicalize --cse

.PHONY: all clean dpu bench bench-single

all: bench dpu

bench: $(BIN_DIR)/bench

# Per-function single benchmark binary (BENCH_FN must be set externally).
# SRC_MLIR is already a single-function file here — copy it directly instead
# of going through the multi-function split-stamp rule.
$(IR_DIR)/$(BENCH_FN).mlir: $(SRC_MLIR) | $(IR_DIR)
	cp $< $@

$(IR_DIR):
	mkdir -p $@

$(BIN_DIR)/$(BENCH_FN): $(IR_DIR)/$(BENCH_FN).dpu.c | $(BIN_DIR)
	$(COMPILE_DPU) $< $@

bench-single: $(IR_DIR)/$(BENCH_FN).o $(BIN_DIR)/$(BENCH_FN) | $(BIN_DIR) $(OUT_DIR)
	clang++ $(CXXFLAGS) -DBENCH_FN=$(BENCH_FN) bench_single.cpp \
	    $(IR_DIR)/$(BENCH_FN).o $(LDFLAGS) \
	    -o $(BIN_DIR)/bench_$(BENCH_FN)

# ─── Split source modules ────────────────────────────────────────────────────

$(IR_DIR)/.split_stamp: $(SRC_MLIR) split_modules.py
	@mkdir -p $(IR_DIR)
	python3 split_modules.py $< $(IR_DIR) $(FUNCS)
	@touch $@

$(IR_DIR)/%.mlir: $(IR_DIR)/.split_stamp ;

# ─── Host compilation ────────────────────────────────────────────────────────

# Prevent Make from deleting intermediate files produced by pattern rules.
.PRECIOUS: $(IR_DIR)/%.kernels.mlir $(IR_DIR)/%.llvm.mlir $(IR_DIR)/%.ll $(IR_DIR)/%.opt.ll

$(IR_DIR)/%.llvm.mlir: $(IR_DIR)/%.mlir
	$(CINM_OPT) $< $(LOWER_PIPELINE) -o $@

$(IR_DIR)/%.ll: $(IR_DIR)/%.llvm.mlir
	$(CINM_TRANS) --allow-unregistered-dialect --mlir-to-llvmir $< -o $@

$(IR_DIR)/%.opt.ll: $(IR_DIR)/%.ll
	$(CLANG_BASE)/opt -O3 -S $< -o $@

$(IR_DIR)/%.o: $(IR_DIR)/%.opt.ll
	$(CLANG_BASE)/llc -O3 -filetype=obj -relocation-model=pic $< -o $@

# ─── DPU kernel compilation ──────────────────────────────────────────────────

$(IR_DIR)/%.kernels.mlir: $(IR_DIR)/%.mlir
	$(CINM_OPT) --upmem-legalize-for-translation $< -o $@

$(IR_DIR)/%.dpu.c: $(IR_DIR)/%.kernels.mlir
	$(CINM_TRANS) --allow-unregistered-dialect --mlir-to-upmem-cpp $< -o $@

dpu: $(FUNCS:%=$(IR_DIR)/%.dpu.c)
	@mkdir -p $(BIN_DIR)
	@for f in $(FUNCS); do \
	    echo "Compiling DPU kernel $$f..."; \
	    $(COMPILE_DPU) $(IR_DIR)/$$f.dpu.c $(BIN_DIR)/$$f; \
	done

# ─── Final host binary ───────────────────────────────────────────────────────

$(BIN_DIR)/bench: bench.cpp $(HOST_OBJS) | $(BIN_DIR) $(OUT_DIR)
	clang++ $(CXXFLAGS) $^ $(LDFLAGS) -o $@

$(BIN_DIR) $(OUT_DIR):
	mkdir -p $@

# ─── Utilities ───────────────────────────────────────────────────────────────

clean:
	rm -rf $(IR_DIR) $(BIN_DIR) $(OUT_DIR)
