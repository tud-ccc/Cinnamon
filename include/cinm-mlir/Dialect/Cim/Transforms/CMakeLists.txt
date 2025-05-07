set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CimTransforms)
add_public_tablegen_target(CimTransformPassesIncGen)

add_mlir_doc(Passes CimTransformPasses ./ -gen-pass-doc)
