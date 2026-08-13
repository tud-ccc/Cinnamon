
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

#endif
