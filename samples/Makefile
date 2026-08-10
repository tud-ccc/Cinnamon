

NAME ?= llama2_110M
# NAME=mha
.DEFAULT_GOAL := $(NAME)_buffered.mlir

.PHONY := clean

clean:
	rm -f $(NAME)_unrolled.mlir $(NAME)_linalg.mlir $(NAME)_with_compute.mlir $(NAME)_buffered.mlir

$(NAME)_unrolled.mlir: $(NAME).mlir
	cinm-opt $< --transform-interpreter > $@

$(NAME)_linalg.mlir: $(NAME).mlir
	cinm-opt $< --inline \
	 --convert-cinm-ops-to-linalg --convert-tensor-to-linalg --linalg-generalize-named-ops --linalg-fuse-elementwise-ops    \
	 --eliminate-empty-tensors --cse \
   > $@

$(NAME)_with_compute.mlir: $(NAME)_linalg.mlir
	cinm-opt $< --cinm-assign-platforms \
	--cinm-isolate-compute-blocks --cinm-expand-compute-scope --cinm-deisolate-compute-blocks \
   > $@


$(NAME)_buffered.mlir: $(NAME)_with_compute.mlir
	cinm-opt $< --cinm-isolate-compute-blocks --eliminate-empty-tensors --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=identity-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops=false --buffer-hoisting --buffer-loop-hoisting --canonicalize --cse --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
