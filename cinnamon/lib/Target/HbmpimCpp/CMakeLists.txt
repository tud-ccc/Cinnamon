add_mlir_translation_library(MLIRHbmpimTargetCpp
  HbmpimTranslateRegistration.cpp
  HbmpimTranslateToCpp.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  HbmpimIR
  MLIRIR
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  )
