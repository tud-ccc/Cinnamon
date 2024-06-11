add_mlir_translation_library(MLIRUPMEMTargetCpp
  UPMEMTranslateRegistration.cpp
  UPMEMTranslateToCpp.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  UPMEMIR
  MLIRIR
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  )
