
set(prefix Cinm)
set(filter cinm)

set(LLVM_TARGET_DEFINITIONS ${prefix}Types.td)
mlir_tablegen(${prefix}Base.h.inc -gen-dialect-decls -dialect=${filter})
mlir_tablegen(${prefix}Base.cpp.inc -gen-dialect-defs -dialect=${filter})
mlir_tablegen(${prefix}Types.h.inc -gen-typedef-decls -typedefs-dialect=${filter})
mlir_tablegen(${prefix}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${filter})

# We generate partial headers as they have dependencies between each other
set(LLVM_TARGET_DEFINITIONS ${prefix}AttributesBase.td)
mlir_tablegen(${prefix}BaseAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})

set(LLVM_TARGET_DEFINITIONS CinmAttributesForOps.td)
mlir_tablegen(${prefix}AttributesForOps.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})

set(LLVM_TARGET_DEFINITIONS ${prefix}PlatformAttrs.td)
mlir_tablegen(${prefix}PlatformAttr.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})

# But a single impl file, as otherwise tablegen duplicates some definitions
set(LLVM_TARGET_DEFINITIONS ${prefix}Attributes.td)
mlir_tablegen(${prefix}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${filter})
mlir_tablegen(${prefix}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${prefix}Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS ${prefix}PlatformAttrInterface.td)
mlir_tablegen(${prefix}PlatformAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(${prefix}PlatformAttrInterface.cpp.inc -gen-attr-interface-defs)


set(LLVM_TARGET_DEFINITIONS ${prefix}Ops.td)
mlir_tablegen(${prefix}Ops.h.inc -gen-op-decls -attrdefs-dialect=${filter})
mlir_tablegen(${prefix}Ops.cpp.inc -gen-op-defs -attrdefs-dialect=${filter})

add_public_tablegen_target(${prefix}IRIncGen)
add_dependencies(${prefix}IncGen ${prefix}IRIncGen)

mlir_gen_iface(Cinm TilingInterface op)
mlir_gen_iface(Cinm CinmComputeOpInterface op)