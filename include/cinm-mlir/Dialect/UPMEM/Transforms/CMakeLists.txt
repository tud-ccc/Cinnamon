set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name UPMEMTransforms)
add_public_tablegen_target(UPMEMTransformPassesIncGen)

add_mlir_doc(Passes UPMEMTransformPasses ./ -gen-pass-doc)
