

PRIMS   = prim_gemv prim_red
TARGETS = cinm2_ca400 \
          cinm2_fast \
          cinm2_hybrid200 cinm2_hybrid400 \
          cinm1 cinm2_ca cinm2_ca200

SOURCES = $(foreach p,$(PRIMS),$(foreach t,$(TARGETS),$(p)_$(t)))

ROOT        := $(shell git -C $(dir $(abspath $(lastword $(MAKEFILE_LIST)))) rev-parse --show-toplevel)
HERE        := $(ROOT)/experiments/paperplots

PREFIX      := prim_largesurrogate
EXP_ROOT    := $(HERE)/data/$(PREFIX)
PLOT_SCRIPT := $(HERE)/plot_best_configs.py
STAMPS_D    := $(EXP_ROOT)/stamps

BO_STAMPS        := $(patsubst %, $(STAMPS_D)/%.bo,           $(SOURCES))
COMPILE_STAMPS   := $(patsubst %, $(STAMPS_D)/%.compiled,     $(SOURCES))
RUN_STAMPS       := $(patsubst %, $(STAMPS_D)/%.run,          $(SOURCES))
AGG_STAMPS       := $(patsubst %, $(STAMPS_D)/%.aggregated,   $(SOURCES))
BO_TIMING_STAMPS := $(patsubst %, $(STAMPS_D)/%.bo_timings,   $(SOURCES))

WORKERS := 64
NICE := 19 # Default is 0, highest is 19

BO_OPTS := --workers $(WORKERS) --n 128 --no-plots --infer-opts n-validation=600 validation-interval=1 max-evals=120 dump-full-pool=false hidden-depth=6  epochs=1810 hidden-width=32

RUN_MAX_CONC := nice -n $(NICE) systemd-run --user --scope -p CPUWeight=20 --same-dir --collect --

# Extract primitive name from a source tag: prim_red_cinm2_hybrid200 → red
prim_of = $(firstword $(subst _, ,$(patsubst prim_%,%,$(1))))

.PHONY: all compile run aggregate bo-timings plots clean bo
.PRECIOUS: $(STAMPS_D)/


all: plots

bo:         $(BO_STAMPS)
compile:    $(COMPILE_STAMPS)
run:        $(RUN_STAMPS)
aggregate:  $(AGG_STAMPS)
bo-timings: $(BO_TIMING_STAMPS)

$(STAMPS_D)/:
	@mkdir -p $@

# $1 = target suffix (e.g. cinm2_hybrid200)
# $2 = simulator
# $3 = eval-timeout-ms
# $4 = extra opts (optional)
define BO_RULE
$(STAMPS_D)/%_$(1).bo: $$(ROOT)/experiments/%.mlir
	@mkdir -p $$(EXP_ROOT)/$$*_$(1)/results
	$$(RUN_MAX_CONC) python $$(ROOT)/experiments/run.py multiseed $$< \
	    --out-dir $$(EXP_ROOT)/$$*_$(1)/results \
			$$(BO_OPTS) simulator=$(2) eval-timeout-ms=$(3) \
			$(4)
	@touch $$@
endef

$(eval $(call BO_RULE,cinm2_ca,cycle-accurate,60000))
$(eval $(call BO_RULE,cinm2_ca200,cycle-accurate,200))
$(eval $(call BO_RULE,cinm2_ca400,cycle-accurate,400))
$(eval $(call BO_RULE,cinm2_fast,fast,60000))
$(eval $(call BO_RULE,cinm2_hybrid200,hybrid,200))
$(eval $(call BO_RULE,cinm2_hybrid400,hybrid,400))
$(eval $(call BO_RULE,cinm1,hybrid,400,use-mram-tiling=false))


# Compile the best config (by cost) for each seed in results dir
$(STAMPS_D)/%.compiled: $(STAMPS_D)/%.bo | $(STAMPS_D)/
	@test -d $(EXP_ROOT)/$*/results || (echo "ERROR: no results dir for $* — run the search first"; exit 1)
	@mkdir -p $(EXP_ROOT)/$*/compiled
	$(RUN_MAX_CONC) python $(HERE)/run_best_configs.py \
		--in-dir $(EXP_ROOT)/$*/results --compile-dir $(EXP_ROOT)/$*/compiled \
		--src $(ROOT)/experiments/prim_$(call prim_of,$*).mlir \
		--prim $(call prim_of,$*) \
		--workers $(WORKERS) \
		--compile-only
	@touch $@

# Benchmark the compiled binaries (--workers 1: serial for accurate wall-clock time)
$(STAMPS_D)/%.run: $(STAMPS_D)/%.compiled | $(STAMPS_D)/
	@mkdir -p $(EXP_ROOT)/$*/run
	python $(HERE)/run_best_configs.py \
		--in-dir $(EXP_ROOT)/$*/results \
		--compile-dir $(EXP_ROOT)/$*/compiled \
    --run-dir $(EXP_ROOT)/$*/run \
		--src $(ROOT)/experiments/prim_$(call prim_of,$*).mlir \
		--prim $(call prim_of,$*) \
		--iters 10 \
		--workers 1 \
		--run-only
	@touch $@

# Aggregate per-config benchmark CSVs into per-type summary CSVs
$(STAMPS_D)/%.aggregated: $(STAMPS_D)/%.run | $(STAMPS_D)/
	@mkdir -p $(EXP_ROOT)/$*/aggregated
	python $(HERE)/aggregate.py \
		--run-dir $(EXP_ROOT)/$*/run --compile-dir $(EXP_ROOT)/$*/compiled --out $(EXP_ROOT)/$*/aggregated
	@touch $@

# Aggregate raw BO-search timings (one CSV per seed) into a single timings.csv
$(STAMPS_D)/%.bo_timings: $(STAMPS_D)/%.bo | $(STAMPS_D)/
	@mkdir -p $(EXP_ROOT)/$*/bo_timings
	@test -d $(EXP_ROOT)/$*/results || (echo "ERROR: no results dir for $* — run the search first"; exit 1)
	python $(HERE)/aggregate_bo_timings.py \
		--results-dir $(EXP_ROOT)/$*/results --out $(EXP_ROOT)/$*/bo_timings
	@touch $@

# Plot violin and bar charts across all sources
plots: $(STAMPS_D)/plots.done

$(STAMPS_D)/plots.done: $(AGG_STAMPS) $(BO_TIMING_STAMPS) | $(STAMPS_D)/
	echo $(SOURCES)
	python $(PLOT_SCRIPT) \
		--exp-root $(EXP_ROOT) \
		--sources $(SOURCES) \
		--out $(EXP_ROOT)/plots/
	@touch $@

clean:
	rm -rf $(STAMPS_D)/
