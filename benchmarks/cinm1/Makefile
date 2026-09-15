

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

ROOT=../..
# opt and llc come from here too: the build symlinks the LLVM tools next to
# its own, from the LLVM it was built against.
CINM_BIN=$(ROOT)/build/bin
CINM_OPT=$(CINM_BIN)/cinm-opt

# The host driver's compiler, settled the same way as in
# cinm_experiments/Makefile and for the same two reasons: a bare `clang++`
# would find the SDK's clang 12, since UPMEM_HOME/bin is on PATH, and $(CXX)
# inside the pixi environment is conda's clang, whose sysroot cannot link the
# SDK's libdpu.so. The system's, overridable.
BENCH_CXX ?= c++
ifneq ($(UPMEM_HOME),)
  ifneq ($(findstring $(UPMEM_HOME),$(BENCH_CXX)),)
    $(error BENCH_CXX is $(BENCH_CXX), inside the UPMEM SDK -- that is its \
            clang 12, not a host compiler. Set BENCH_CXX.)
  endif
endif

# openblas and numa are declared in pixi.toml; without the explicit -I/-L the
# system copies win even inside the environment.
ifneq ($(CONDA_PREFIX),)
  CONDA_CXXFLAGS = -isystem $(CONDA_PREFIX)/include
  CONDA_LDFLAGS  = -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

$(IR_PREFIX).1.tiled.mlir: $(BENCH_NAME).mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --pass-pipeline="builtin.module(func.func(cinm-infer-tile-sizes,cinm-tiling,cinm-isolate-compute-blocks))" | $(CINM_OPT) --pass-pipeline="builtin.module(**cinm.compute_block(affine-loop-unroll{unroll-full-threshold=1},canonicalize),cinm-deisolate-compute-blocks)" > $@

# $(IR_PREFIX).1a.cinm.buffered.mlir: $(IR_PREFIX).1.tiled.mlir
# 	$(CINM_OPT) $< --eliminate-empty-tensors --canonicalize --cse --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map'  \
# 					--cse --canonicalize --buffer-results-to-out-params=hoist-static-allocs > $@

# $(IR_PREFIX).1b.cnm.mlir: $(IR_PREFIX).1a.cinm.buffered.mlir
# 	$(CINM_OPT) $< --convert-cinm-to-cnm --canonicalize --cnm-hoist-workgroups --canonicalize --cse > $@

$(IR_PREFIX).2.cnm.mlir: $(IR_PREFIX).1.tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --canonicalize --cnm-hoist-workgroups --canonicalize --cse > $@

#					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \

# $(IR_PREFIX).1c.cnm.affine.mlir: $(IR_PREFIX).1b.cnm.mlir
# 	$(CINM_OPT) $< --eliminate-empty-tensors  \
# 					--cse --canonicalize --buffer-results-to-out-params=hoist-static-allocs \
# 				  --convert-linalg-to-affine-loops --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse > $@

# $(IR_PREFIX).3.cnm.affine.mlir: $(IR_PREFIX).1c.cnm.affine.mlir

$(IR_PREFIX).3.cnm.affine.mlir: $(IR_PREFIX).2.cnm.mlir
	$(CINM_OPT) $< --eliminate-empty-tensors --cse \
	   			--one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
					--cse --canonicalize --convert-linalg-to-affine-loops --buffer-loop-hoisting --buffer-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs \
				   --canonicalize --cse > $@

$(IR_PREFIX).4.cnm.after_opts.mlir: $(IR_PREFIX).3.cnm.affine.mlir
	$(CINM_OPT) $< \
					--promote-buffers-to-stack --fold-memref-alias-ops --canonicalize --affine-loop-fusion --sroa --canonicalize --affine-scalrep --loop-invariant-code-motion --affine-loop-invariant-code-motion --sroa --affine-scalrep \
				   --canonicalize --cse > $@

#	$(CINM_OPT) $< --affine-loop-unroll=unroll-factor=4 --lower-affine --buffer-loop-hoisting --buffer-hoisting

$(IR_PREFIX).5.cnm.scf.mlir: $(IR_PREFIX).4.cnm.after_opts.mlir
	$(CINM_OPT) $<  --lower-affine --buffer-loop-hoisting --buffer-hoisting \
				   --canonicalize --cse > $@

$(IR_PREFIX).6.upmem.mlir: $(IR_PREFIX).5.cnm.scf.mlir
	$(CINM_OPT) $< --cnm-ensure-scatter-gather-contiguous --convert-cnm-to-upmem --cse --upmem-dedup-kernels --cse  > $@

$(IR_PREFIX).6a1.llvm.mlir: $(IR_PREFIX).6.upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize --cinm-unwrap-compute-blocks \
	--convert-upmem-to-llvm --convert-scf-to-cf \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true \
  --expand-strided-metadata --memref-expand --lower-affine --convert-to-llvm \
	 --cse --reconcile-unrealized-casts --canonicalize --llvm-legalize-for-export --canonicalize --cse \
	> $@
# --fold-memref-alias-ops \

$(IR_PREFIX).6b.dpu.c: $(IR_PREFIX).6.upmem.mlir
	$(CINM_BIN)/cinm-translate --allow-unregistered-dialect --mlir-to-upmem-cpp $< > $@

$(IR_PREFIX).6a2.host.ll: $(IR_PREFIX).6a1.llvm.mlir
	$(CINM_BIN)/cinm-translate --allow-unregistered-dialect --mlir-to-llvmir $< > $@

$(IR_PREFIX).6a3.host.opt.ll: $(IR_PREFIX).6a2.host.ll
	$(CINM_BIN)/opt -O3 -S $< > $@

$(IR_PREFIX).6a3a.host.dbg.o: $(IR_PREFIX).6a2.host.ll
	$(CINM_BIN)/llc $< -o $@ -filetype=obj -relocation-model=pic

$(IR_PREFIX).6a3b.host.o: $(IR_PREFIX).6a3.host.opt.ll
	$(CINM_BIN)/llc -O3 $< -o $@ -filetype=obj -relocation-model=pic

# Compile all DPU kernel executables
$(BENCH_NAME)-kernel-compile: $(IR_PREFIX).6b.dpu.c
	$(CINM_BIN)/cinm-compile-dpu $< $(OUT_DIR)/bin

# HOST executable
$(OUT_DIR)/bin/host: $(IR_PREFIX).6a3b.host.o $(BENCH_NAME).cpp
	@mkdir -p $(@D)
	# openblas backs common.hpp's golden references and numa its output
	# buffers; BENCH_FN only names the stats files common.hpp can write,
	# which these drivers do not use, so the module name will do.
	$(BENCH_CXX) -g -std=c++17 -DBENCH_FN=$(BENCH_NAME) $^ -lUpmemDialectRuntime -fPIE -ldpu -ldpuverbose \
	    -lopenblas -lnuma -L${UPMEM_HOME}/lib -L$(ROOT)/build/lib \
	    -I${UPMEM_HOME}/include/dpu $(CONDA_CXXFLAGS) $(CONDA_LDFLAGS) -o $@

$(BENCH_NAME): $(IR_PREFIX).6a3.host.opt.ll $(IR_PREFIX).6b.dpu.c

$(BENCH_NAME)-exe: $(OUT_DIR)/bin/host $(BENCH_NAME)-kernel-compile

clean:
	rm -rf $(OUT_DIR)/irs/* $(OUT_DIR)/bin/*
