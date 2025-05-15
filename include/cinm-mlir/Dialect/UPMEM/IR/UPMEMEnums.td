//===- Attributes.td - UPMEM dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_ENUMS
#define UPMEM_ENUMS

include "cinm-mlir/Dialect/UPMEM/IR/UPMEMBase.td"
include "mlir/IR/EnumAttr.td"

def UPMEM_MemcpyOpMRAMToWRAM: I32EnumAttrCase<"MRAMToWRAM", 0, "mram_to_wram">;
def UPMEM_MemcpyOpWRAMToMRAM: I32EnumAttrCase<"WRAMToMRAM", 1, "wram_to_mram">;

def MemcpyDir : I32EnumAttr<"TransferDirection",
  "copy direction between wram and mram", [
    UPMEM_MemcpyOpMRAMToWRAM,
    UPMEM_MemcpyOpWRAMToMRAM
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::upmem";
}

def MemcpyDirAttr : EnumAttr<UPMEM_Dialect, MemcpyDir,
                                  "transfer_dir">;

#endif
