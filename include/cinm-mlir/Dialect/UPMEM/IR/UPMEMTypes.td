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
include "cinm-mlir/Dialect/Cinm/IR/CinmWorkgroupTypeInterface.td"

def DeviceHierarchyType : UPMEM_Type<"DeviceHierarchy", [CinmWorkgroupTypeInterface]> {
  let mnemonic = "hierarchy";
  let summary = "A set of allocated DPUs and the tasklets running on them";
  let description = [{
    `!upmem.hierarchy<DxT>`: D DPUs running T tasklets each. The SDK
    allocates DPU *counts* -- which ranks they land on is its business, and
    unobservable -- so ranks are deliberately not part of the shape.
  }];

  let parameters = (ins "int32_t":$numDpus, "int32_t":$numTaskletsPerDpu);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    int64_t getNumElements() const {
      return getNumDpus() * getNumTaskletsPerDpu();
    }

    int64_t getNumDims() const { return 2;}

    SmallVector<int64_t, 2> getWgShape() const {
      return {getNumDpus(), getNumTaskletsPerDpu()};
    }

    /// cinm::WorkgroupTypeInterface -- what lets a hierarchy allocated
    /// outside a compute block be recognized as a forwarded workgroup by
    /// code that cannot name this dialect.
    ::llvm::SmallVector<int64_t> getWorkgroupShape() const {
      return {getNumDpus(), getNumTaskletsPerDpu()};
    }

  }];
}


#endif
