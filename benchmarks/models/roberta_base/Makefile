# The whole-program front end: source -> unrolled -> linalg -> with_compute
# -> buffered. Interactive twin of the `wholeprog_front` doit task, which
# drives this same makefile (with CINM_OPT and OUT set) so the pipeline is
# written down exactly once.

# Passed by doit so the binary built for the campaign is the one that runs;
# a bare `make` takes whatever is on PATH.
CINM_OPT ?= cinm-opt
# Where the intermediates land. `.` keeps a bare `make` building in place.
OUT ?= .

NAME ?= roberta_base
# NAME=roberta_base
.DEFAULT_GOAL := $(OUT)/$(NAME)_buffered.mlir

# A cinm-opt that dies mid-module still leaves the redirect's truncated
# output behind, and make would take that stub as up to date.
.DELETE_ON_ERROR:

.PHONY := clean

clean:
	rm -f $(OUT)/$(NAME)_unrolled.mlir $(OUT)/$(NAME)_linalg.mlir $(OUT)/$(NAME)_with_compute.mlir $(OUT)/$(NAME)_buffered.mlir

# The schedule is preloaded from a sidecar so the payload never carries
# transform ops: cinm-translate does not register that dialect, and a module
# that reaches host code generation with its schedule still inside fails
# there. The sidecar is a prerequisite so a schedule edit rebuilds.
$(OUT)/$(NAME)_unrolled.mlir: $(NAME).mlir $(NAME).schedule.mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --transform-preload-library=transform-library-paths=$(NAME).schedule.mlir --transform-interpreter > $@

$(OUT)/$(NAME)_linalg.mlir: $(OUT)/$(NAME)_unrolled.mlir
	$(CINM_OPT) $< --inline \
	 --convert-cinm-ops-to-linalg --convert-tensor-to-linalg --linalg-generalize-named-ops \
	 --linalg-fuse-elementwise-ops  \
	 --eliminate-empty-tensors --cse \
   > $@

# Offload decision. require-profitable turns on the roofline gate, so an op
# reaches the device only if the platform predicts it beats the host there;
# everything else stays a plain host op. This is the experiment for this
# model, so the gate is on here and nowhere in the operator suites. The host
# figure is the int8 one -- Cascade Lake's VNNI does 64 int8 MACs per
# instruction, four times the int32 rate the default assumes -- because both
# sides of this comparison run the quantized model.
ASSIGN_OPTS ?= require-profitable=true host-ops-per-second=8.6e12

$(OUT)/$(NAME)_with_compute.mlir: $(OUT)/$(NAME)_linalg.mlir
	$(CINM_OPT) $< "--cinm-assign-platforms=$(ASSIGN_OPTS)" \
	--cinm-isolate-compute-blocks --cinm-expand-compute-scope --cinm-deisolate-compute-blocks --cinm-complete-compute-graph \
   > $@


$(OUT)/$(NAME)_buffered.mlir: $(OUT)/$(NAME)_with_compute.mlir
	$(CINM_OPT) $< --cnm-isolate-linalg-captures --cinm-isolate-compute-blocks --cinm-simplify-compute-signatures --eliminate-empty-tensors --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=identity-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops=false --buffer-loop-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
