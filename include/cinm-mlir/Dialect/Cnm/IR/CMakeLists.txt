
set(prefix Cnm)
set(filter cnm)

set(LLVM_TARGET_DEFINITIONS ${prefix}Interfaces.td)
mlir_tablegen(${prefix}Interfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(${prefix}Interfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(${prefix}TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(${prefix}TypeInterfaces.cpp.inc -gen-type-interface-defs)

mlir_gen_ir(Cnm)
