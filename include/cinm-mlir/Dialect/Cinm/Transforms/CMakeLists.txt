set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CinmTransforms)
add_public_tablegen_target(CinmTransformPassesIncGen)

add_mlir_doc(Passes CinmTransformPasses ./ -gen-pass-doc)
