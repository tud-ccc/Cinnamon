set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HbmpimTransforms)
add_public_tablegen_target(HbmpimTransformPassesIncGen)

add_mlir_doc(Passes HbmpimTransformPasses ./ -gen-pass-doc)
