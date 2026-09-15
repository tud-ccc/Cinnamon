//===- CinmComputeOpInterface.td ----------------------*- tablegen -*-===//
//
// This is the definitions file for the CinmComputeOpInterface.
//
//===------------------------------------------------------------------===//

#ifndef CINM_COMPUTE_OP_INTERFACE
#define CINM_COMPUTE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def CinmComputeOpInterface : OpInterface<"ComputeOpInterface"> {
    let cppNamespace = "::mlir::cinm";

    let description = [{
      Interface for ops that delimit a region of code for offloading
      (`cinm.compute` and `cinm.compute_block`). Provides access to
      the optional accelerator attribute that describes the hardware
      target.
    }];

    let methods = [
        InterfaceMethod<
            "Returns the optional accelerator attribute.",
            "::std::optional<::mlir::cinm::CinmAcceleratorAttrInterface>",
            "getAccelerator",
            (ins),
            "",
            "">
    ];

    let extraClassDeclaration = [{
      template<class T>
      std::optional<T> getAcceleratorAs() {
        auto ax = getAccelerator();
        T res;
        if (!ax || !(res = llvm::dyn_cast_or_null<T>(*ax))) return std::nullopt;
        return res;
      }
    }];
}

#endif
