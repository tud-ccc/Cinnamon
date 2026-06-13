//===- Attributes.td - UPMEM dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_ATTRIBUTES
#define UPMEM_ATTRIBUTES

include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.td"

// def UPMEM_AddressSpaceGlobal : I32EnumAttrCase<"Global", 1, "global">;
// def UPMEM_AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 2, "workgroup">;
// def UPMEM_AddressSpacePrivate : I32EnumAttrCase<"Private", 3, "private">;
// def UPMEM_AddressSpaceEnum : UPMEM_I32Enum<
//   "AddressSpace", "UPMEM address space", [
//     UPMEM_AddressSpaceGlobal,
//     UPMEM_AddressSpaceWorkgroup,
//     UPMEM_AddressSpacePrivate
//   ]>;

// def UPMEM_AddressSpaceAttr :
//   UPMEM_I32EnumAttr<"address_space", UPMEM_AddressSpaceEnum>;

def UPMEM_SimulatorAttr : I64EnumAttr<
    "SimulatorKind", "",
    [
      I64EnumAttrCase<"OPCOUNT", 0, "opcount">,
      I64EnumAttrCase<"CA", 1, "cycleaccurate">
    ]> {
  let cppNamespace = "::mlir::upmem";
  let genSpecializedAttr = 0;
}

def DpuMemSpace : I64EnumAttr<
    "DpuMemSpace", "",
    [
      I64EnumAttrCase<"MRAM", 0, "mram">,
      I64EnumAttrCase<"WRAM", 1, "wram">
    ]> {
  let cppNamespace = "::mlir::upmem";
  let genSpecializedAttr = 0;
}

def UPMEM_DpuMemSpaceAttr : EnumAttr<UPMEM_Dialect, DpuMemSpace, ""> {
  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;
}

include "UPMEMTileFirstAttributes.td"

#endif
