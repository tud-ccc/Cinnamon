
set(prefix UPMEM)
set(filter upmem)

set(LLVM_TARGET_DEFINITIONS ${prefix}Ops.td)

mlir_tablegen(${prefix}Base.h.inc -gen-dialect-decls -dialect=${filter})
mlir_tablegen(${prefix}Base.cpp.inc -gen-dialect-defs -dialect=${filter})
mlir_tablegen(${prefix}Types.h.inc -gen-typedef-decls -typedefs-dialect=${filter})
mlir_tablegen(${prefix}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${filter})
mlir_tablegen(${prefix}Ops.h.inc -gen-op-decls -dialect=${filter})
mlir_tablegen(${prefix}Ops.cpp.inc -gen-op-defs -dialect=${filter})
# Gen only the cpp from the ops
mlir_tablegen(${prefix}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${filter})

set(LLVM_TARGET_DEFINITIONS ${prefix}Enums.td)
mlir_tablegen(${prefix}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${prefix}Enums.cpp.inc -gen-enum-defs)
mlir_tablegen(${prefix}EnumAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})

set(LLVM_TARGET_DEFINITIONS ${prefix}TileFirstAttributes.td)
mlir_tablegen(${prefix}T1Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})

add_public_tablegen_target(${prefix}IRIncGen)
add_dependencies(${prefix}IncGen ${prefix}IRIncGen)
    
