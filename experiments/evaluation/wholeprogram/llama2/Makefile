# The whole-program front end: source -> unrolled -> linalg -> with_compute
# -> buffered. Interactive twin of the `wholeprog_front` doit task, which
# drives this same makefile (with CINM_OPT and OUT set) so the pipeline is
# written down exactly once.

# Passed by doit so the binary built for the campaign is the one that runs;
# a bare `make` takes whatever is on PATH.
CINM_OPT ?= cinm-opt
# Where the intermediates land. `.` keeps a bare `make` building in place.
OUT ?= .

NAME ?= llama2_110M
# NAME=mha
.DEFAULT_GOAL := $(OUT)/$(NAME)_buffered.mlir

# A cinm-opt that dies mid-module still leaves the redirect's truncated
# output behind, and make would take that stub as up to date.
.DELETE_ON_ERROR:

.PHONY := clean

clean:
	rm -f $(OUT)/$(NAME)_unrolled.mlir $(OUT)/$(NAME)_linalg.mlir $(OUT)/$(NAME)_with_compute.mlir $(OUT)/$(NAME)_buffered.mlir

$(OUT)/$(NAME)_unrolled.mlir: $(NAME).mlir
	@mkdir -p $(@D)
	$(CINM_OPT) $< --transform-interpreter > $@

$(OUT)/$(NAME)_linalg.mlir: $(OUT)/$(NAME)_unrolled.mlir
	$(CINM_OPT) $< --inline \
	 --convert-cinm-ops-to-linalg --convert-tensor-to-linalg --linalg-generalize-named-ops \
	 --linalg-fuse-elementwise-ops  \
	 --eliminate-empty-tensors --cse \
   > $@

$(OUT)/$(NAME)_with_compute.mlir: $(OUT)/$(NAME)_linalg.mlir
	$(CINM_OPT) $< --cinm-assign-platforms \
	--cinm-isolate-compute-blocks --cinm-expand-compute-scope --cinm-deisolate-compute-blocks --cinm-complete-compute-graph \
   > $@


$(OUT)/$(NAME)_buffered.mlir: $(OUT)/$(NAME)_with_compute.mlir
	$(CINM_OPT) $< --cnm-isolate-linalg-captures --cinm-isolate-compute-blocks --cinm-simplify-compute-signatures --eliminate-empty-tensors --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=identity-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops=false --buffer-loop-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
