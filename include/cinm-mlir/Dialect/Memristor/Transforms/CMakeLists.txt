set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MemristorTransforms)
add_public_tablegen_target(MemristorTransformPassesIncGen)

add_mlir_doc(Passes MemristorTransformPasses ./ -gen-pass-doc)
