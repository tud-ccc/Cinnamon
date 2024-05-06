//===- TilingInterface.td -----------------------------*- tablegen -*-===//
//
// This is the definitions file for the cinm TilingInterface.
//
//===-----------------------------------------------------------------===//

#ifndef CINM_TILING_INTERFACE
#define CINM_TILING_INTERFACE

include "mlir/IR/OpBase.td"

def CinmTilingInterface : OpInterface<"CinmTilingInterface"> {
    let cppNamespace = "::mlir::cinm";

    let description = [{
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/ [{
            }],
            /*retTy=*/ "Value",
            /*methodName=*/ "tile",
            /*args=*/ (ins "OpBuilder":$builder, "ArrayRef<int64_t>":$tileSizes),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ "">];
}

#endif
