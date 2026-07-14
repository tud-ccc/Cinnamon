

PRIMS   = prim_gemv prim_red
TARGETS = cinm2_ca cinm2_ca200 cinm2_ca400 \
          cinm2_fast \
          cinm2_hybrid200 cinm2_hybrid400 \
          cinm1

SOURCES = $(foreach p,$(PRIMS),$(foreach t,$(TARGETS),$(p)_$(t)))

ROOT        := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
HERE        := $(ROOT)/experiments/paperplots

EXP_ROOT    := $(ROOT)/.experiments/tags
PLOT_SCRIPT := $(HERE)/plot_best_configs.py

BO_STAMPS        := $(patsubst %, .stamps/%.bo,           $(SOURCES))
RUN_STAMPS       := $(patsubst %, .stamps/%.run,          $(SOURCES))
AGG_STAMPS       := $(patsubst %, .stamps/%.aggregated,   $(SOURCES))
BO_TIMING_STAMPS := $(patsubst %, .stamps/%.bo_timings,   $(SOURCES))

WORKERS := 32
BO_OPTS := --workers $(WORKERS) --n 128 --no-plots --out-dir @+results --infer-opts n-validation=600 validation-interval=1 max-evals=120

# Extract primitive name from a source tag: prim_red_cinm2_hybrid200 → red
prim_of = $(firstword $(subst _, ,$(patsubst prim_%,%,$(1))))

.PHONY: all run aggregate bo-timings plots clean all-bo
.PRECIOUS: .stamps/


all: plots

bo:         $(BO_STAMPS)
run:        $(RUN_STAMPS)
aggregate:  $(AGG_STAMPS)
bo-timings: $(BO_TIMING_STAMPS)

.stamps/:
	@mkdir -p $@

.stamps/%_cinm2_ca.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_ca python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=cycle-accurate eval-timeout-ms=60000
	@touch $@

.stamps/%_cinm2_ca200.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_ca200 python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=cycle-accurate eval-timeout-ms=200
	@touch $@

.stamps/%_cinm2_ca400.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_ca400 python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=cycle-accurate eval-timeout-ms=400
	@touch $@

.stamps/%_cinm2_fast.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_fast python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=fast eval-timeout-ms=0
	@touch $@

.stamps/%_cinm2_hybrid200.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_hybrid200 python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=hybrid eval-timeout-ms=200
	@touch $@

.stamps/%_cinm2_hybrid400.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm2_hybrid400 python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=hybrid eval-timeout-ms=400
	@touch $@

.stamps/%_cinm1.bo: $(ROOT)/experiments/%.mlir
	exp run +$*_cinm1 python $(ROOT)/experiments/run.py multiseed $< $(BO_OPTS) simulator=hybrid eval-timeout-ms=300 use-mram-tiling=false
	@touch $@


# Compile and benchmark the best config (by cost) for each seed under @results
.stamps/%.run: .stamps/%.bo | .stamps/
	@test -d $(EXP_ROOT)/$*/results || (echo "ERROR: no results handle for $* — run the search first"; exit 1)
	exp run +$* python $(HERE)/run_best_configs.py --workers 48 --iters 10 \
		--in-dir @results --run-dir @+run \
		--src $(ROOT)/experiments/prim_$(call prim_of,$*).mlir \
		--prim $(call prim_of,$*)
	@touch $@

# Aggregate per-config benchmark CSVs into per-type summary CSVs
.stamps/%.aggregated: .stamps/%.run | .stamps/
	exp run +$* python $(ROOT)/experiments/upmemcm/reduce_cost/aggregate.py \
		--run-dir @run --out @+aggregated
	@touch $@

# Aggregate raw BO-search timings (one CSV per seed) into a single timings.csv
.stamps/%.bo_timings: .stamps/%.bo | .stamps/
	@test -d $(EXP_ROOT)/$*/results || (echo "ERROR: no results handle for $* — run the search first"; exit 1)
	exp run +$* python $(HERE)/aggregate_bo_timings.py \
		--results-dir @results --out @+bo_timings
	@touch $@

# Plot violin and bar charts across all sources
plots: .stamps/plots.done

.stamps/plots.done: $(AGG_STAMPS) $(BO_TIMING_STAMPS) | .stamps/
	python $(PLOT_SCRIPT) \
		--exp-root $(EXP_ROOT) \
		--sources $(SOURCES) \
		--out $(HERE)/plots/
	@touch $@

clean:
	rm -rf .stamps/
