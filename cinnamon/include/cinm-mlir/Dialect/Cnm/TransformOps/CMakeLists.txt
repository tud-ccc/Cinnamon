
set(LLVM_TARGET_DEFINITIONS CnmTransformOps.td)
mlir_tablegen(CnmTransformOps.h.inc -gen-op-decls)
mlir_tablegen(CnmTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCnmTransformOpsIncGen)
add_dependencies(CnmIncGen MLIRCnmTransformOpsIncGen)


set(LLVM_TARGET_DEFINITIONS TransformPass.td)
mlir_tablegen(TransformPass.h.inc -gen-pass-decls -name CnmTransformInterpreter)
add_public_tablegen_target(CnmTransformInterpreterPassIncGen)
add_dependencies(CnmIncGen CnmTransformInterpreterPassIncGen)
add_mlir_doc(Passes CnmTransformInterpreter ./ -gen-pass-doc)
