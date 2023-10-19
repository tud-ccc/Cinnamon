set(LLVM_TARGET_DEFINITIONS CinmPasses.td)
mlir_tablegen(CinmPasses.h.inc -gen-pass-decls -name CinmConversion)
add_public_tablegen_target(CinmConversionPassIncGen)

add_mlir_doc(CinmPasses CinmConversionPasses ./ -gen-pass-doc)


