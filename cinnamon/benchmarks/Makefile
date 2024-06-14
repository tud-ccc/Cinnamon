

BENCH_NAME=va
.DEFAULT_GOAL := $(BENCH_NAME)
.PHONY: $(BENCH_NAME) clean

CINM_OPT=../build/bin/cinm-opt

# Add optimization passes to this target 
$(BENCH_NAME).tiled.mlir: $(BENCH_NAME).mlir
	$(CINM_OPT) $< --cinm-tiling --cse > $@

$(BENCH_NAME).cnm.mlir: $(BENCH_NAME).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --func-bufferize --lower-affine --scf-bufferize  --tensor-bufferize --arith-bufferize --bufferization-bufferize --cse > $@

$(BENCH_NAME).upmem.mlir: $(BENCH_NAME).cnm.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --upmem-outline-kernel --cse > $@

$(BENCH_NAME).llvm.mlir: $(BENCH_NAME).upmem.mlir
	$(CINM_OPT) $< --convert-upmem-to-llvm --mlir-print-ir-after-failure  --canonicalize \
	--convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm \
    --fold-memref-alias-ops --expand-strided-metadata --memref-expand --finalize-memref-to-llvm \
	--convert-func-to-llvm=use-bare-ptr-memref-call-conv=true --cse \
	> $@

$(BENCH_NAME).upmem.c: $(BENCH_NAME).upmem.mlir
	../build/bin/cinm-translate --mlir-to-upmem-cpp $< > $@

$(BENCH_NAME): $(BENCH_NAME).llvm.mlir $(BENCH_NAME).upmem.c

clean:
	rm $(BENCH_NAME).*.mlir $(BENCH_NAME).upmem.c
