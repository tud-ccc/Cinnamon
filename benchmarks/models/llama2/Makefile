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

# Which ops the accelerator is a candidate for. The roofline gate this pass
# can apply (require-profitable) is left off: it prices an op against the
# whole array, which for a small op is the wrong end of the curve, and it
# runs before any search space exists. What decides placement is the menu
# screen inside --upmem-infer-accelerator, which makes the same comparison
# against every DPU count the block admits. The verdicts are still dumped
# here, as what the whole-array model would have said.
#
# The host figure is the int8 one -- Cascade Lake's VNNI does 64 int8 MACs
# per instruction, four times the int32 rate the default assumes -- because
# both sides of the E2E comparison run quantized.
ifneq (,$(filter $(NAME),$(E2E_MODELS)))
ASSIGN_OPTS ?= host-ops-per-second=8.6e12
endif
ASSIGN_OPTS ?=
# Every verdict the gate took, with its roofline terms, as JSON lines: the
# input of the evaluation's roofline figure. Written for the ungated models
# too, where it records what the gate would have said.
DECISIONS = $(OUT)/$(NAME)_offload_decisions.jsonl
ASSIGN_PASS = --cinm-assign-platforms=$(strip $(ASSIGN_OPTS) dump-decisions=$(DECISIONS))

# A cinm-opt that dies mid-module still leaves the redirect's truncated
# output behind, and make would take that stub as up to date.
.DELETE_ON_ERROR:

# Every stage is a pass pipeline spelled out below, so a stage rebuilds when
# this file changes, not only when its input does.
MK := $(lastword $(MAKEFILE_LIST))

.PHONY := clean

clean:
	rm -f $(OUT)/$(NAME)_unrolled.mlir $(OUT)/$(NAME)_linalg.mlir $(OUT)/$(NAME)_with_compute.mlir $(OUT)/$(NAME)_offload_decisions.jsonl $(OUT)/$(NAME)_device.mlir $(OUT)/$(NAME)_host.mlir $(OUT)/$(NAME)_buffered.mlir

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

# The program split for a host compiler: _host.mlir holds the bodies of the
# host blocks as functions (what IREE compiles), _device.mlir the rest, where
# each host block calls its function through a prototype. Host blocks that
# only views separate are merged first, so each stretch of host code between
# two device blocks is one function.
#
# SPLIT_SRC is which module to split. The default is this front end's last
# stage, where every block the accelerator is a candidate for is still a
# device block. What the program actually runs is decided later, by the menu
# screen inside --upmem-infer-accelerator, and the blocks it leaves on the
# host are host code that wants merging like any other -- so the evaluation
# points SPLIT_SRC at the module that stage stamps.
SPLIT_SRC ?= $(OUT)/$(NAME)_with_compute.mlir

$(OUT)/$(NAME)_device.mlir $(OUT)/$(NAME)_host.mlir &: $(SPLIT_SRC) $(MK)
	$(CINM_OPT) $< --cinm-merge-host-blocks --cinm-outline-host-blocks=outlined-file=$(OUT)/$(NAME)_host.mlir > $(OUT)/$(NAME)_device.mlir


$(OUT)/$(NAME)_buffered.mlir: $(OUT)/$(NAME)_with_compute.mlir $(MK)
	$(CINM_OPT) $< --cnm-isolate-linalg-captures --cinm-isolate-compute-blocks --cinm-simplify-compute-signatures --eliminate-empty-tensors --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=identity-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops=false --buffer-loop-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
