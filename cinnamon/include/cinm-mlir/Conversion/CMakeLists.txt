
set(TBLGEN_EXTRA $<$<BOOL:${CINM_BUILD_GPU_SUPPORT}>:-DCINM_GPU_SUPPORT>)

set(LLVM_TARGET_DEFINITIONS CinmPasses.td)
mlir_tablegen(CinmPasses.h.inc -gen-pass-decls -name CinmConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(CinmConversionPassIncGen)

add_mlir_doc(CinmPasses CinmConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS CimPasses.td)
mlir_tablegen(CimPasses.h.inc -gen-pass-decls -name CimConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(CimConversionPassIncGen)

add_mlir_doc(CimPasses CimConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS CnmPasses.td)
mlir_tablegen(CnmPasses.h.inc -gen-pass-decls -name CnmConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(CnmConversionPassIncGen)

add_mlir_doc(CnmPasses CnmConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS MemristorPasses.td)
mlir_tablegen(MemristorPasses.h.inc -gen-pass-decls -name MemristorConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(MemristorConversionPassIncGen)

add_mlir_doc(MemristorPasses MemristorConversionPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS UPMEMPasses.td)
mlir_tablegen(UPMEMPasses.h.inc -gen-pass-decls -name UPMEMConversion ${TBLGEN_EXTRA})
add_public_tablegen_target(UPMEMConversionPassIncGen)

add_mlir_doc(UPMEMPasses UPMEMConversionPasses ./ -gen-pass-doc)


