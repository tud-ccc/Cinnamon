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
M_VALUES := 16777216
#4194304 16777216
# T: chunk size in elements (= bytes copied per loop iteration / 4)
T_VALUES := 8 16 32 64 128 256 1024 4096 16384 65536

CONFIGS    := $(foreach m,$(M_VALUES),$(foreach t,$(T_VALUES),$(m)_$(t)))
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

# ── Helpers to extract M / T from a "M_T" config name ──────────────────────
cfg_M = $(word 1,$(subst _, ,$1))
cfg_T = $(word 2,$(subst _, ,$1))

.PHONY: all run clean

all: $(BENCH_BINS)

# Write header once, then let each binary append its rows.
run: all | $(OUT_DIR)
	printf "M,T,time_iter_ns\n" > $(RESULT_CSV)
	for b in $(BENCH_BINS); do $$b $(RESULT_CSV); done
	@echo "Results written to $(RESULT_CSV)"

$(IR_DIR) $(BIN_DIR) $(OUT_DIR):
	mkdir -p $@

# ── Per-config rules (template instantiation + final link) ──────────────────

define BENCH_RULES
$(IR_DIR)/bench_$(1).mlir: bench.mlir.tpl instantiate.py | $(IR_DIR)
	python3 instantiate.py bench.mlir.tpl \
	    --M=$(call cfg_M,$(1)) --T=$(call cfg_T,$(1)) -o $$@

$(BIN_DIR)/bench_$(1): $(IR_DIR)/bench_$(1).o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) \
	    -DM_VAL=$(call cfg_M,$(1)) -DT_VAL=$(call cfg_T,$(1)) \
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
