#[========================================================================[.rst:
MLIRUtils
---------

Some utility functions for reliably declaring MLIR TableGen targets and
enforcing a common naming scheme.

#]========================================================================]

function(mlir_gen_enums prefix)
    set(LLVM_TARGET_DEFINITIONS Enums.td)

    mlir_tablegen(Enums.h.inc -gen-enum-decls)
    mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

    add_public_tablegen_target(${prefix}EnumsIncGen)
    add_dependencies(${prefix}IncGen ${prefix}EnumsIncGen)
endfunction()

function(mlir_gen_iface prefix iface kind)
    set(LLVM_TARGET_DEFINITIONS ${iface}.td)

    mlir_tablegen(${iface}.h.inc -gen-${kind}-interface-decls)
    mlir_tablegen(${iface}.cpp.inc -gen-${kind}-interface-defs)

    add_public_tablegen_target(${prefix}${iface}InterfaceIncGen)
    add_dependencies(${prefix}IncGen ${prefix}${iface}InterfaceIncGen)
endfunction()

function(mlir_gen_ir prefix)
    string(TOLOWER ${prefix} filter)

    set(LLVM_TARGET_DEFINITIONS ${prefix}Ops.td)

    mlir_tablegen(${prefix}Base.h.inc -gen-dialect-decls -dialect=${filter})
    mlir_tablegen(${prefix}Base.cpp.inc -gen-dialect-defs -dialect=${filter})
    mlir_tablegen(${prefix}Types.h.inc -gen-typedef-decls -typedefs-dialect=${filter})
    mlir_tablegen(${prefix}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${filter})
    mlir_tablegen(${prefix}Enums.h.inc -gen-enum-decls)
    mlir_tablegen(${prefix}Enums.cpp.inc -gen-enum-defs)
    mlir_tablegen(${prefix}Attributes.h.inc -gen-attrdef-decls -attrdefs-dialect=${filter})
    mlir_tablegen(${prefix}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${filter})
    mlir_tablegen(${prefix}Ops.h.inc -gen-op-decls -dialect=${filter})
    mlir_tablegen(${prefix}Ops.cpp.inc -gen-op-defs -dialect=${filter})

    add_public_tablegen_target(${prefix}IRIncGen)
    add_dependencies(${prefix}IncGen ${prefix}IRIncGen)
endfunction()
