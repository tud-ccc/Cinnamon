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
              Tile this operation using the given tiling factors.  
            }],
            /*retTy=*/ "::mlir::DiagnosedSilenceableFailure",
            /*methodName=*/ "convertToTiledOps",
            /*args=*/ (ins "RewriterBase&":$rewriter, "ArrayRef<int64_t>":$tilingFactors, "SmallVectorImpl<Value>&":$results),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ "">];
}

#endif
