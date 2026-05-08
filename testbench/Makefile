

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

$(IR_PREFIX).1.tiled.mlir: $(BENCH_NAME).mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --pass-pipeline="builtin.module(func.func(cinm-infer-tile-sizes,cinm-tiling))" | $(CINM_OPT) --pass-pipeline="builtin.module(**cinm.compute(affine-loop-unroll{unroll-full unroll-full-threshold=1},canonicalize))" > $@

$(IR_PREFIX).1a.cinm.buffered.mlir: $(IR_PREFIX).1.tiled.mlir
	$(CINM_OPT) $< --eliminate-empty-tensors --cse --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'  \
					--cse --canonicalize --buffer-results-to-out-params=hoist-static-allocs > $@

$(IR_PREFIX).1b.cnm.mlir: $(IR_PREFIX).1a.cinm.buffered.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --cnm-hoist-workgroups --canonicalize --cse > $@

$(IR_PREFIX).2.cnm.mlir: $(IR_PREFIX).1.tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --cnm-hoist-workgroups --canonicalize --cse > $@

#					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \

$(IR_PREFIX).1c.cnm.affine.mlir: $(IR_PREFIX).1b.cnm.mlir
	$(CINM_OPT) $< --eliminate-empty-tensors  \
					--cse --canonicalize --buffer-results-to-out-params=hoist-static-allocs \
				  --convert-linalg-to-affine-loops --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse > $@

$(IR_PREFIX).3.cnm.affine.mlir: $(IR_PREFIX).2.cnm.mlir
	$(CINM_OPT) $< --eliminate-empty-tensors --cse --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'  \
					--cse --canonicalize --convert-linalg-to-affine-loops --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs \
				   --canonicalize --cse > $@

$(IR_PREFIX).4.cnm.after_opts.mlir: $(IR_PREFIX).3.cnm.affine.mlir
	$(CINM_OPT) $< \
					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \
				   --canonicalize --cse > $@

$(IR_PREFIX).5.cnm.scf.mlir: $(IR_PREFIX).4.cnm.after_opts.mlir
	$(CINM_OPT) $< --lower-affine --buffer-loop-hoisting --buffer-hoisting \
				   --canonicalize --cse > $@

$(IR_PREFIX).6.upmem.mlir: $(IR_PREFIX).5.cnm.scf.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --cse --upmem-dedup-kernels --cse > $@

$(IR_PREFIX).6a1.llvm.mlir: $(IR_PREFIX).6.upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize --cinm-unwrap-compute-blocks \
	--convert-upmem-to-llvm --convert-scf-to-cf \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
  --expand-strided-metadata --memref-expand --lower-affine --convert-to-llvm \
	 --cse --reconcile-unrealized-casts --canonicalize --llvm-legalize-for-export --canonicalize --cse \
	> $@
# --fold-memref-alias-ops \

$(IR_PREFIX).6b.dpu.c: $(IR_PREFIX).6.upmem.mlir
	../build/bin/cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp $< > $@

$(IR_PREFIX).6a2.host.ll: $(IR_PREFIX).6a1.llvm.mlir
	../build/bin/cinm-translate --allow-unregistered-dialect --mlir-to-llvmir $< > $@

$(IR_PREFIX).6a3.host.opt.ll: $(IR_PREFIX).6a2.host.ll
	$(CLANG_18_BASE)/opt -O3 -S $< > $@

$(IR_PREFIX).6a3a.host.dbg.o: $(IR_PREFIX).6a2.host.ll
	$(CLANG_18_BASE)/llc $< -o $@ -filetype=obj -relocation-model=pic

$(IR_PREFIX).6a3b.host.o: $(IR_PREFIX).6a3.host.opt.ll
	$(CLANG_18_BASE)/llc -O3 $< -o $@ -filetype=obj -relocation-model=pic

# Compile all DPU kernel executables
$(BENCH_NAME)-kernel-compile: $(IR_PREFIX).6b.dpu.c
	lib/compile_dpu.sh $< $(OUT_DIR)/bin

# HOST executable
$(OUT_DIR)/bin/host: $(IR_PREFIX).6a3b.host.o
	@mkdir -p $(@D)
	clang++ -g apps/$(BENCH_NAME).cpp $(IR_PREFIX).6a3b.host.o -lUpmemDialectRuntime -fPIE -ldpu -ldpuverbose -L${UPMEM_HOME}/lib -L../build/lib -I${UPMEM_HOME}/include/dpu -o $@

$(BENCH_NAME): $(IR_PREFIX).6a3.host.opt.ll $(IR_PREFIX).6b.dpu.c

$(BENCH_NAME)-exe: $(OUT_DIR)/bin/host $(BENCH_NAME)-kernel-compile

clean:
	rm -rf $(OUT_DIR)/irs/* $(OUT_DIR)/bin/*
