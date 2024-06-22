

BENCH_NAME ?= va
.DEFAULT_GOAL := $(BENCH_NAME)
.PHONY: $(BENCH_NAME) $(BENCH_NAME)-kernel-compile clean

OUT_BASE=generated2
OUT_DIR=$(OUT_BASE)/$(BENCH_NAME)
OUT_DIR_IR=$(OUT_DIR)/irs
OUT_DIR_BIN=$(OUT_DIR)/bin
OUT_DIR_HOST=$(OUT_DIR)/host
OUT_DIR_DPU=$(OUT_DIR)/dpu

IR_PREFIX=$(OUT_DIR)/irs/$(BENCH_NAME)
BUILD_PREFIX=$(OUT_DIR)/out

CLANG_18_BASE=../../llvm/build/bin
CINM_OPT=../build/bin/cinm-opt

$(IR_PREFIX).tiled.mlir: $(BENCH_NAME).mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --cinm-tiling --affine-loop-unroll='unroll-full unroll-full-threshold=1' > $@

$(IR_PREFIX).cnm.mlir: $(IR_PREFIX).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --cnm-hoist-workgroups --canonicalize --cse > $@

$(IR_PREFIX).cnm.bufferized.mlir: $(IR_PREFIX).cnm.mlir
	$(CINM_OPT) $< --lower-affine --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'  \
				  --convert-linalg-to-affine-loops --lower-affine --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse > $@

$(IR_PREFIX).upmem.mlir: $(IR_PREFIX).cnm.bufferized.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --cse --upmem-outline-kernel --upmem-dedup-kernels --cse > $@

$(IR_PREFIX).llvm.mlir: $(IR_PREFIX).upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize \
	--convert-scf-to-cf --convert-cf-to-llvm --fold-memref-alias-ops --lower-affine --convert-arith-to-llvm \
	--convert-upmem-to-llvm \
    --expand-strided-metadata --memref-expand --finalize-memref-to-llvm --lower-affine --convert-arith-to-llvm \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true --cse --reconcile-unrealized-casts --llvm-legalize-for-export --canonicalize --cse \
	> $@

$(IR_PREFIX).dpu.c: $(IR_PREFIX).upmem.mlir
	../build/bin/cinm-translate --mlir-to-upmem-cpp $< > $@

$(IR_PREFIX).host.ll: $(IR_PREFIX).llvm.mlir
	../build/bin/cinm-translate --mlir-to-llvmir $< > $@

$(IR_PREFIX).host.opt.ll: $(IR_PREFIX).host.ll
	$(CLANG_18_BASE)/opt -O3 -S $< > $@

$(IR_PREFIX).host.o: $(IR_PREFIX).host.opt.ll
	$(CLANG_18_BASE)/llc -O3 $< -o $@ -filetype=obj -relocation-model=pic

# Compile all DPU kernel executables
$(BENCH_NAME)-kernel-compile: $(IR_PREFIX).dpu.c
	lib/compile_dpu.sh $< $(OUT_DIR)/bin

# HOST executable
$(OUT_DIR)/bin/host: $(IR_PREFIX).host.o $(IR_PREFIX).dpu.c
	@mkdir -p $(@D)
	clang++ -g apps/$(BENCH_NAME).cpp $(IR_PREFIX).host.o -lUpmemDialectRuntime -fPIE -ldpu -ldpuverbose -L${UPMEM_HOME}/lib -L../build/lib -I${UPMEM_HOME}/include/dpu -o $@

# $(BENCH_NAME): $(OUT_DIR)/bin/host $(OUT_DIR)/bin/dpu
$(BENCH_NAME): $(IR_PREFIX).host.opt.ll $(IR_PREFIX).dpu.c

$(BENCH_NAME)-exe: $(OUT_DIR)/bin/host $(BENCH_NAME)-kernel-compile

clean:
	rm -rf $(OUT_DIR)
