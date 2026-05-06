

NAME=llama2_110M
# NAME=mha

$(NAME)_linalg.mlir: $(NAME).mlir
	cinm-opt $< --inline \
	 --convert-cinm-ops-to-linalg --convert-tensor-to-linalg --linalg-generalize-named-ops --linalg-fuse-elementwise-ops --linalg-inline-scalar-operands --cse   \
	 --eliminate-empty-tensors --cse \
   > $@

$(NAME)_buffered.mlir: $(NAME)_linalg.mlir
	cinm-opt $< --cse --canonicalize --one-shot-bufferize=function-boundary-type-conversion=infer-layout-map\ bufferize-function-boundaries\ allow-return-allocs-from-loops --buffer-hoisting --canonicalize --buffer-results-to-out-params=hoist-static-allocs --cse --canonicalize > $@
