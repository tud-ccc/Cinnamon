set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AlpineTransforms)
add_public_tablegen_target(AlpineTransformPassesIncGen)

add_mlir_doc(Passes AlpineTransformPasses ./ -gen-pass-doc)
