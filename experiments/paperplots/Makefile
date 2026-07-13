FUNCS   = red_4MB red_64MB red_256MB red_512MB
#SOURCES = prim_red_cinm2_CA prim_red_cinm2_CA_200ms prim_red_cinm2_fast \
          prim_red_cinm2_hybrid200 prim_red_cinm2_hybrid400
SOURCES = prim_red_cinm2_hybrid200 prim_red_cinm2_fast

ROOT        := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
HERE        := $(ROOT)/experiments/paperplots

EXP_ROOT    := $(ROOT)/.experiments/tags
SRC_MLIR    := $(ROOT)/experiments/prim_red.mlir
PLOT_SCRIPT := $(HERE)/plot_best_configs.py

RUN_STAMPS  := $(patsubst %, .stamps/%.run,        $(SOURCES))
AGG_STAMPS  := $(patsubst %, .stamps/%.aggregated, $(SOURCES))

.PHONY: all run aggregate plots clean
.PRECIOUS: .stamps/

all: plots

run:       $(RUN_STAMPS)
aggregate: $(AGG_STAMPS)

.stamps/:
	@mkdir -p $@

# Compile and benchmark the best config (by cost) for each seed under @results
.stamps/%.run: | .stamps/
	@test -d $(EXP_ROOT)/$*/results || (echo "ERROR: no results handle for $* — run the search first"; exit 1)
	exp run +$* python $(HERE)/run_best_configs.py --workers 32 --iters 10 \
		--in-dir @results --run-dir @+run --src $(SRC_MLIR)
	@touch $@

# Aggregate per-config benchmark CSVs into per-type summary CSVs
.stamps/%.aggregated: .stamps/%.run | .stamps/
	exp run +$* python $(ROOT)/experiments/upmemcm/reduce_cost/aggregate.py \
		--run-dir @run --out @+aggregated
	@touch $@

# Plot violin and bar charts across all sources
plots: .stamps/plots.done

.stamps/plots.done: $(AGG_STAMPS) | .stamps/
	python $(PLOT_SCRIPT) \
		--exp-root $(EXP_ROOT) \
		--sources $(SOURCES) \
		--out $(HERE)/plots/
	@touch $@

clean:
	rm -rf .stamps/
