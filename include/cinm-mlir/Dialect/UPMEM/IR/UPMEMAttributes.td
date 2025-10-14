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


def UPMEM_MemcpyOpMRAMToWRAM: I32EnumAttrCase<"MRAMToWRAM", 0, "mram_to_wram">;
def UPMEM_MemcpyOpWRAMToMRAM: I32EnumAttrCase<"WRAMToMRAM", 1, "wram_to_mram">;

def MemcpyDir : I32EnumAttr<"MemcpyDirOp",
  "copy operation between wram and mram", [
    UPMEM_MemcpyOpMRAMToWRAM,
    UPMEM_MemcpyOpWRAMToMRAM
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::upmem";
}

def MemcpyDirAttr : EnumAttr<UPMEM_Dialect, MemcpyDir,
                                  "memcpy_dir">;



#endif
