# The whole-program front end: source -> unrolled -> linalg -> with_compute
# -> buffered. Interactive twin of the `wholeprog_front` doit task, which
# drives this same makefile (with CINM_OPT and OUT set) so the pipeline is
# written down exactly once.

# Passed by doit so the binary built for the campaign is the one that runs;
# a bare `make` takes whatever is on PATH.
CINM_OPT ?= cinm-opt
# Where the intermediates land. `.` keeps a bare `make` building in place.
OUT ?= .

# llama2_110M and llama2_110M_fusedqkv are the f32 models of the RQ4 stacks;
# llama2_110M_w8a8, llama_3B_w8a8 and llama2_7B_w8a8 are the integer decode
# models of the E2E subsection.
NAME ?= llama2_110M
E2E_MODELS := llama2_110M_w8a8 llama_3B_w8a8 llama2_7B_w8a8
.DEFAULT_GOAL := $(OUT)/$(NAME)_buffered.mlir

# The schedule is preloaded from a sidecar so the payload never carries
# transform ops: cinm-translate does not register that dialect, and a module
# that reaches host code generation with its schedule still inside fails
# there. The two f32 models share one schedule.
SCHEDULE ?= $(NAME).schedule.mlir
ifeq ($(NAME),llama2_110M_fusedqkv)
SCHEDULE := llama2_110M.schedule.mlir
endif

# Offload decision. require-profitable turns on the roofline gate, so an op
# reaches the device only if the platform predicts it beats the host there;
# everything else stays a plain host op. That is the experiment for the
# integer model, so the gate is on for it and nowhere else: the f32 models
# are the RQ4 stacks' programs, whose arms compare allocations of a fully
# offloaded graph. The host figure is the int8 one -- Cascade Lake's VNNI
# does 64 int8 MACs per instruction, four times the int32 rate the default
# assumes -- because both sides of the E2E comparison run quantized.
ifneq (,$(filter $(NAME),$(E2E_MODELS)))
ASSIGN_OPTS ?= require-profitable=true host-ops-per-second=8.6e12
endif
ASSIGN_OPTS ?=
ASSIGN_PASS = $(if $(ASSIGN_OPTS),--cinm-assign-platforms=$(ASSIGN_OPTS),--cinm-assign-platforms)

# A cinm-opt that dies mid-module still leaves the redirect's truncated
# output behind, and make would take that stub as up to date.
.DELETE_ON_ERROR:

# Every stage is a pass pipeline spelled out below, so a stage rebuilds when
# this file changes, not only when its input does.
MK := $(lastword $(MAKEFILE_LIST))

.PHONY := clean

clean:
	rm -f $(OUT)/$(NAME)_unrolled.mlir $(OUT)/$(NAME)_linalg.mlir $(OUT)/$(NAME)_with_compute.mlir $(OUT)/$(NAME)_buffered.mlir

$(OUT)/$(NAME)_unrolled.mlir: $(NAME).mlir $(SCHEDULE) $(MK)
	@mkdir -p $(@D)
	$(CINM_OPT) $< --transform-preload-library=transform-library-paths=$(SCHEDULE) --transform-interpreter > $@

$(OUT)/$(NAME)_linalg.mlir: $(OUT)/$(NAME)_unrolled.mlir $(MK)
	$(CINM_OPT) $< --inline \
	 --convert-cinm-ops-to-linalg --convert-tensor-to-linalg --linalg-generalize-named-ops \
	 --linalg-fuse-elementwise-ops=fuse-with-recompute=false \
	 --eliminate-empty-tensors --cse \
   > $@

$(OUT)/$(NAME)_with_compute.mlir: $(OUT)/$(NAME)_linalg.mlir $(MK)
	$(CINM_OPT) $< "$(ASSIGN_PASS)" --cinm-absorb-static-slices \
	--cinm-isolate-compute-blocks --cinm-expand-compute-scope --cinm-deisolate-compute-blocks --cinm-complete-compute-graph \
   > $@


$(OUT)/$(NAME)_buffered.mlir: $(OUT)/$(NAME)_with_compute.mlir $(MK)
	$(CINM_OPT) $< --cnm-isolate-linalg-captures --cinm-isolate-compute-blocks --cinm-simplify-compute-signatures --eliminate-empty-tensors --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=identity-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops=false --buffer-loop-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
