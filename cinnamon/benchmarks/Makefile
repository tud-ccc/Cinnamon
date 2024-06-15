

BENCH_NAME ?= va
.DEFAULT_GOAL := $(BENCH_NAME)
.PHONY: $(BENCH_NAME) clean

OUT_DIR=generated2/$(BENCH_NAME)
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
	$(CINM_OPT) $< --cinm-tiling --cse > $@

$(IR_PREFIX).cnm.mlir: $(IR_PREFIX).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --func-bufferize --lower-affine --scf-bufferize  --tensor-bufferize --arith-bufferize --bufferization-bufferize --cse > $@

$(IR_PREFIX).upmem.mlir: $(IR_PREFIX).cnm.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --convert-linalg-to-affine-loops --lower-affine --upmem-outline-kernel --cse > $@

$(IR_PREFIX).llvm.mlir: $(IR_PREFIX).upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize \
	--convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
	--convert-upmem-to-llvm \
    --fold-memref-alias-ops --expand-strided-metadata --memref-expand --finalize-memref-to-llvm \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true --cse --reconcile-unrealized-casts --cse \
	> $@

$(IR_PREFIX).dpu.c: $(IR_PREFIX).upmem.mlir
	../build/bin/cinm-translate --mlir-to-upmem-cpp $< > $@

$(IR_PREFIX).host.ll: $(IR_PREFIX).llvm.mlir
	../build/bin/cinm-translate --mlir-to-llvmir $< > $@

$(IR_PREFIX).host.opt.ll: $(IR_PREFIX).host.ll
	$(CLANG_18_BASE)/opt -O3 -S $< > $@

$(IR_PREFIX).host.o: $(IR_PREFIX).host.opt.ll
	$(CLANG_18_BASE)/llc -O3 $< -o $@ -filetype=obj -relocation-model=pic

hostlib.o:
	clang lib/host/host_lib.c -I${UPMEM_HOME}/include/dpu -c -o $@

# DPU kernel executable
$(OUT_DIR)/bin/dpu: $(IR_PREFIX).dpu.c
	@mkdir -p $(@D)
	cp lib/dpu/dpu_lib.h $(OUT_DIR)/irs
	clang $< -o $@

# HOST executable
$(OUT_DIR)/bin/host: $(IR_PREFIX).host.o $(IR_PREFIX).dpu.c hostlib.o
	@mkdir -p $(@D)
	clang++ apps/$(BENCH_NAME).cpp $(IR_PREFIX).host.o hostlib.o -fPIE -ldpu -ldpuverbose -L${UPMEM_HOME}/lib -I${UPMEM_HOME}/include/dpu -o $@

$(BENCH_NAME): $(OUT_DIR)/bin/host $(OUT_DIR)/bin/dpu

clean:
	rm -rf $(OUT_DIR) hostlib.o
