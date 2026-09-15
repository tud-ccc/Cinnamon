//===- Attributes.td - UPMEM dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_ATTRIBUTES
#define UPMEM_ATTRIBUTES

include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td"

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

def DpuMemSpace : I64EnumAttr<
    "DpuMemSpace", "",
    [
      I64EnumAttrCase<"MRAM", 0, "mram">,
      I64EnumAttrCase<"WRAM", 1, "wram">
    ]> {
  let cppNamespace = "::mlir::upmem";
  let genSpecializedAttr = 0;
}

def UPMEM_DpuMemSpaceAttr : EnumAttr<UPMEM_Dialect, DpuMemSpace, "",
    [CinmLevelAttrInterface]> {
  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // The stringified enum cases are exactly the names `upmemLevels()` gives
    // the corresponding CinmLevelDefAttrs, so a memory space round-trips to
    // its level descriptor through UpmemPlatformAttr::getLevel.
    llvm::StringRef getLevelName() const {
      return stringifyDpuMemSpace(getValue());
    }
  }];
}

include "UPMEMTileFirstAttributes.td"

#endif
