set(LLVM_TARGET_DEFINITIONS ${dialectNameUpper}Passes.td)
mlir_tablegen(${dialectNameUpper}Passes.h.inc -gen-pass-decls -name ${dialectNameUpper}Conversion)
add_public_tablegen_target(${dialectNameUpper}ConversionPassIncGen)

add_mlir_doc(${dialectNameUpper}Passes ${dialectNameUpper}ConversionPasses ./ -gen-pass-doc)


