set(LLVM_TARGET_DEFINITIONS CinmPasses.td)
mlir_tablegen(CinmPasses.h.inc -gen-pass-decls -name CinmConversion)
add_public_tablegen_target(CinmConversionPassIncGen)

add_mlir_doc(CinmPasses CinmConversionPasses ./ -gen-pass-doc)


set(LLVM_TARGET_DEFINITIONS CnmPasses.td)
mlir_tablegen(CnmPasses.h.inc -gen-pass-decls -name CnmConversion)
add_public_tablegen_target(CnmConversionPassIncGen)

add_mlir_doc(CnmPasses CnmConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS UPMEMPasses.td)
mlir_tablegen(UPMEMPasses.h.inc -gen-pass-decls -name UPMEMConversion)
add_public_tablegen_target(UPMEMConversionPassIncGen)

add_mlir_doc(UPMEMPasses UPMEMConversionPasses ./ -gen-pass-doc)


