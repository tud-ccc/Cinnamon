
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

    // InterfaceMethod<
    //   /*desc=*/"Make progress on scheduling in the given block.",
    //   /*retTy=*/"::mlir::DiagnosedSilenceableFailure",
    //   /*methodName=*/"makeSchedulingProgress",
    //   /*args=*/(ins "::mlir::cinm::CinmAcceleratorScheduler&":$api)
    // >,
    // InterfaceMethod<
    //   /*desc=*/"Return a parameterized memory space attr for the given level",
    //   /*retTy=*/"::mlir::cinm::CinmMemSpaceAttr",
    //   /*methodName=*/"getGenericMemSpace",
    //   /*args=*/(ins "::mlir::cinm::CinmLevelDefAttr":$level)
    // >
  ];

  let extraClassDeclaration = [{

    // Defined out-of-line in CnmAttributes.h after WorkgroupType is complete.
    mlir::cnm::WorkgroupType getWorkgroupType() const;

    // /// Return whether a given memory space belongs to this accelerator.
    // bool ownsMemSpace(::mlir::cinm::CinmMemSpaceAttr space) const {
    //   if (space.isAbsentMemspace() || space.isHostMemspace()) 
    //       return false;

    //   auto level = getPlatform().getLevel(space.getName());
    //   if (!level)
    //     return false;
    
    //   return space == getGenericMemSpace(level);
    // }

    // /// Return whether all the operands of the given op (minus those of unknown memspace) are buffers belonging to this accelerator.
    // bool ownsOperation(::mlir::Operation* op) const {
    //   for (auto operand : op->getOperands()) {
    //     if (auto buf = llvm::dyn_cast_or_null<::mlir::cinm::CinmBufferType>(operand.getType())) {
    //       auto space = buf.getMemorySpace();
    //       if (space.isAbsentMemspace())
    //         continue;
    //       if (!ownsMemSpace(space))
    //         return false;
    //     }
    //   }
    //   return true;
    // }
  }];
}


#endif
