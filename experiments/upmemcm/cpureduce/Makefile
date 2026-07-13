ROOT        = ../../..
CINM_OPT    = $(ROOT)/build/bin/cinm-opt
CINM_TRANS  = $(ROOT)/build/bin/cinm-translate
CLANG_BASE  = $(ROOT)/third-party/llvm/build/bin
CXX         = clang++
CXXFLAGS    = -O3 -std=c++17

IR_DIR  = ir
BIN_DIR = bin
OUT_DIR = output

RESULT_CSV = $(OUT_DIR)/results.csv

# ── Sweep parameters ────────────────────────────────────────────────────────
# M: total array size in i32 elements (4MB / 16MB / 64MB of data)
M_VALUES := 8 16 32 64 128 256 512 1024 2048 4096

CONFIGS    := $(foreach m,$(M_VALUES),$(m))
BENCH_BINS := $(CONFIGS:%=$(BIN_DIR)/bench_%)

# ── MLIR lowering pipeline (pure affine/memref/func → LLVM) ────────────────
LOWER_PIPELINE = \
    --convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
    --expand-strided-metadata \
    --memref-expand \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-to-llvm \
    --cse --reconcile-unrealized-casts --canonicalize \
    --llvm-legalize-for-export --canonicalize --cse

.PHONY: all run clean

all: $(BENCH_BINS)

# Write header once, then let each binary append its rows.
run: all | $(OUT_DIR)
	printf "M,time_iter_ns\n" > $(RESULT_CSV)
	for b in $(BENCH_BINS); do $$b $(RESULT_CSV); done
	@echo "Results written to $(RESULT_CSV)"

$(IR_DIR) $(BIN_DIR) $(OUT_DIR):
	mkdir -p $@

# ── Per-config rules (template instantiation + final link) ──────────────────

define BENCH_RULES
$(IR_DIR)/bench_$(1).mlir: bench.mlir.tpl instantiate.py | $(IR_DIR)
	python3 instantiate.py bench.mlir.tpl \
	    --M=$(1) -o $$@

$(BIN_DIR)/bench_$(1): $(IR_DIR)/bench_$(1).o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) \
	    -DM_VAL=$(1) \
	    bench.cpp $(IR_DIR)/bench_$(1).o -o $$@
endef

$(foreach cfg,$(CONFIGS),$(eval $(call BENCH_RULES,$(cfg))))

# ── MLIR → LLVM IR → object ─────────────────────────────────────────────────

$(IR_DIR)/%.llvm.mlir: $(IR_DIR)/%.mlir
	$(CINM_OPT) $< $(LOWER_PIPELINE) -o $@

$(IR_DIR)/%.ll: $(IR_DIR)/%.llvm.mlir
	$(CINM_TRANS) --allow-unregistered-dialect --mlir-to-llvmir $< -o $@

$(IR_DIR)/%.opt.ll: $(IR_DIR)/%.ll
	$(CLANG_BASE)/opt -O3 -S $< -o $@

$(IR_DIR)/%.o: $(IR_DIR)/%.opt.ll
	$(CLANG_BASE)/llc -O3 -filetype=obj -relocation-model=pic $< -o $@

# ── Utilities ────────────────────────────────────────────────────────────────

clean:
	rm -rf $(IR_DIR) $(BIN_DIR) $(OUT_DIR)
