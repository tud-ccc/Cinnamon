//===- Types.td - UPMEM dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the UPMEM dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef UPMEM_TYPES
#define UPMEM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "UPMEMBase.td"

def DeviceHierarchyType : UPMEM_Type<"DeviceHierarchy"> {
  let mnemonic = "hierarchy";
  let summary = "";

  let parameters = (ins "int32_t":$numRanks, "int32_t":$numDpusPerRank, "int32_t":$numTaskletsPerDpu);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    int64_t getNumElements() const {
      return getNumRanks() * getNumDpusPerRank() * getNumTaskletsPerDpu();
    }

    int64_t getNumDims() const { return 3;}

    SmallVector<int64_t, 3> getWgShape() const {
      return {getNumRanks(), getNumDpusPerRank(), getNumTaskletsPerDpu()};
    }

  }];
}


#endif
