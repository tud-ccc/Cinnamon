set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CnmTransforms)
add_public_tablegen_target(CnmTransformPassesIncGen)

add_mlir_doc(Passes CnmTransformPasses ./ -gen-pass-doc)
