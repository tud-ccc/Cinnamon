

BENCH_NAME=va
.DEFAULT_GOAL := $(BENCH_NAME)
.PHONY: $(BENCH_NAME) clean

CLANG_18_BASE=../../llvm/build/bin

CINM_OPT=../build/bin/cinm-opt

$(BENCH_NAME).tiled.mlir: $(BENCH_NAME).mlir
	$(CINM_OPT) $< --cinm-tiling --cse > $@

$(BENCH_NAME).cnm.mlir: $(BENCH_NAME).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --func-bufferize --lower-affine --scf-bufferize  --tensor-bufferize --arith-bufferize --bufferization-bufferize --cse > $@

$(BENCH_NAME).upmem.mlir: $(BENCH_NAME).cnm.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --convert-linalg-to-affine-loops --lower-affine --upmem-outline-kernel --cse > $@

$(BENCH_NAME).llvm.mlir: $(BENCH_NAME).upmem.mlir
	$(CINM_OPT) $< --mlir-print-ir-after-failure --canonicalize \
	--convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
	--convert-upmem-to-llvm \
    --fold-memref-alias-ops --expand-strided-metadata --memref-expand --finalize-memref-to-llvm \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true --cse --reconcile-unrealized-casts --cse \
	> $@

$(BENCH_NAME).upmem.c: $(BENCH_NAME).upmem.mlir
	../build/bin/cinm-translate --mlir-to-upmem-cpp $< > $@

$(BENCH_NAME).host.ll: $(BENCH_NAME).llvm.mlir
	../build/bin/cinm-translate --mlir-to-llvmir $< | opt -O3 -S > $@

$(BENCH_NAME).host.o: $(BENCH_NAME).host.ll
	$(CLANG_18_BASE)/llc $< -o $@ -filetype=obj

hostlib.o:
	clang lib/host/host_lib.c -I../../upmem/include/dpu -ldpu -ldpuverbose -c

$(BENCH_NAME): $(BENCH_NAME).host.o $(BENCH_NAME).upmem.c hostlib.o
	clang++ apps/$(BENCH_NAME).cpp $(BENCH_NAME).host.o hostlib.o -ldpu -L${UPMEM_HOME}/lib -I${UPMEM_HOME}/include/dpu

clean:
	rm $(BENCH_NAME).*.mlir $(BENCH_NAME).upmem.c $(BENCH_NAME).*.ll *.o *.s
