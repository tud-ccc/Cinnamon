

BENCH_NAME ?= va
.DEFAULT_GOAL := $(BENCH_NAME)
.PHONY: $(BENCH_NAME) $(BENCH_NAME)-kernel-compile clean

OUT_BASE=gen
OUT_DIR=$(OUT_BASE)/$(BENCH_NAME)
OUT_DIR_IR=$(OUT_DIR)/irs
OUT_DIR_BIN=$(OUT_DIR)/bin
OUT_DIR_HOST=$(OUT_DIR)/host
OUT_DIR_DPU=$(OUT_DIR)/dpu

IR_PREFIX=$(OUT_DIR)/irs/$(BENCH_NAME)
BUILD_PREFIX=$(OUT_DIR)/out

CLANG_18_BASE=../third-party/llvm/build/bin
CINM_OPT=../build/bin/cinm-opt

$(IR_PREFIX).tiled.mlir: $(BENCH_NAME).mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --pass-pipeline="builtin.module(func.func(cinm-isolate-compute-blocks,cinm-tiling))" | $(CINM_OPT) --pass-pipeline="builtin.module(**cinm.compute(affine-loop-unroll{unroll-full unroll-full-threshold=1},canonicalize))" > $@

$(IR_PREFIX).cnm.mlir: $(IR_PREFIX).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --cnm-hoist-workgroups --canonicalize --cse > $@

#					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \

$(IR_PREFIX).cnm.affine.mlir: $(IR_PREFIX).cnm.mlir
	$(CINM_OPT) $< --lower-affine --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'  \
					--canonicalize --buffer-results-to-out-params=hoist-static-allocs \
				  --convert-linalg-to-affine-loops --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse > $@

$(IR_PREFIX).cnm.after_opts.mlir: $(IR_PREFIX).cnm.affine.mlir
	$(CINM_OPT) $< \
					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \
				   --canonicalize --cse > $@

$(IR_PREFIX).cnm.scf.mlir: $(IR_PREFIX).cnm.after_opts.mlir
	$(CINM_OPT) $< --lower-affine --buffer-loop-hoisting --buffer-hoisting \
				   --canonicalize --cse > $@

$(IR_PREFIX).upmem.mlir: $(IR_PREFIX).cnm.scf.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --cse --upmem-dedup-kernels --cse > $@

$(IR_PREFIX).llvm.mlir: $(IR_PREFIX).upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize \
	--convert-upmem-to-llvm --convert-scf-to-cf \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
  --expand-strided-metadata --memref-expand --lower-affine --convert-to-llvm \
	 --cse --reconcile-unrealized-casts --canonicalize --llvm-legalize-for-export --canonicalize --cse \
	> $@
# --fold-memref-alias-ops \

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
