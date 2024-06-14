

.DEFAULT_GOAL := va

CINM_OPT=../build/bin/cinm-opt
BENCH_NAME="$(BENCH_NAME).

# Add optimization passes to this target 
$(BENCH_NAME).tiled.mlir: $(BENCH_NAME).mlir
	$(CINM_OPT) $< --cinm-tiling --cse > $@

$(BENCH_NAME).cnm.mlir: $(BENCH_NAME).tiled.mlir
	$(CINM_OPT) $< --convert-cinm-to-cnm --func-bufferize --lower-affine --scf-bufferize  --tensor-bufferize --arith-bufferize --cse > $@

$(BENCH_NAME).upmem.mlir: $(BENCH_NAME).cnm.mlir
	$(CINM_OPT) $< --convert-cnm-to-upmem --upmem-outline-kernel --cse > $@

$(BENCH_NAME).llvm.mlir: $(BENCH_NAME).upmem.mlir
	$(CINM_OPT) $< --convert-upmem-to-llvm --cse > $@

$(BENCH_NAME).upmem.c: $(BENCH_NAME).upmem.mlir
	../build/bin/cinm-translate --mlir-to-upmem-cpp $< > $@


clean:
	rm $(BENCH_NAME).*.mlir $(BENCH_NAME).upmem.c