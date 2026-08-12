
#ifndef CNM_INTERFACES
#define CNM_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td"

def CnmAcceleratorAttrInterface : AttrInterface<"CnmAcceleratorAttrInterface", [
  CinmAcceleratorAttrInterface
]> {
  let description = [{
    Interface for attributes that can be used as an accelerator declaration for CNM.
  }];

  let cppNamespace = "::mlir::cnm";

  let methods = [
    InterfaceMethod<
      /*desc=*/
      "Return the shape of this workgroup in CNM format.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getWorkgroupShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/
      "Return buffer capacity descriptors for each workgroup level.",
      /*retTy=*/"::llvm::SmallVector<::mlir::cinm::CinmLevelArrayAttr>",
      /*methodName=*/"getWorkgroupMemoryLevels",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*desc=*/"Return the size in bytes of the memory that leaves of the workgroup can use. If implemented, this method allows the CINM->CNM pass to determine automatically a mapping from host (cinm) op to leaf (eg tasklets). This mapping is fallible, we keep it for backwards compatibility. In the future we will remove this to make the cinm->cnm transformation more granular using transform ops.",
      /*retTy=*/"int64_t",
      /*methodName=*/"bufferSizeOfLeaf",
      /*args=*/(ins),
      "",
      "return 0;"
    >,

  ];

  let extraClassDeclaration = [{

    // Defined out-of-line in CnmAttributes.h after WorkgroupType is complete.
    mlir::cnm::WorkgroupType getWorkgroupType() const;

  }];
}

def CnmWorkgroupTypeInterface : TypeInterface<"CnmWorkgroupTypeInterface"> {
  let description = [{
    A handle to the piece of an accelerator a program runs on: a grid of
    leaves, of some shape. `!cnm.workgroup` implements it, and so do the
    target-level handles it lowers into (`!upmem.hierarchy`).

    The interface exists for workgroups that cross a region boundary: a
    whole-program schedule allocates a group's device set ONCE, outside the
    compute blocks that use it -- necessarily at the allocator's own dialect
    level -- and forwards the value into each member block. Target-agnostic
    code (verifiers, the graph-level allocation stage) must recognize such a
    forwarded value and read its shape without naming any target dialect,
    and this interface is exactly that contract. See
    docs/EvaluationImplementationPlan.md sec. 3.2.
  }];

  let cppNamespace = "::mlir::cnm";

  let methods = [
    InterfaceMethod<
      /*desc=*/
      "Return the shape of the workgroup this value is a handle to, one "
      "extent per level of the hierarchy (for UPMEM: DPUs, then tasklets).",
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Number of leaves of the workgroup (product of the shape).",
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumLeaves",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        int64_t product = 1;
        for (int64_t extent : $_type.getWorkgroupShape())
          product *= extent;
        return product;
      }]
    >,
  ];
}


#endif
