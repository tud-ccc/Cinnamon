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


def UPMEM_AsyncToken : UPMEM_Type<"AsyncToken"> {
  let mnemonic = "async.token";
  let summary = "";
}

def DeviceHierarchy : UPMEM_Type<"DeviceHierarchy"> {
  let mnemonic = "hierarchy";
  let summary = "";

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t getNumElements() {
      auto wgShape = getShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
    int64_t getNumDims() { return getShape().size();}

    int64_t getNumRanks() {
      return getShape()[0];
    }
    int64_t getNumDpusPerRank() {
      return getShape()[1];
    }
    int64_t getNumTaskletsPerDpu() {
      return getShape()[2];
    }
  }];
}


#endif
