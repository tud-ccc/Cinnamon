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
      Interface for Cinm operations that can be tiled down into a loop nest
      around a smaller instance of the op.

      Tiling can be applied by the `--cinm-tiling` pass. For that to work it
      needs tileable ops to be tagged with an appropriate `cinm.tile_sizes`
      attribute. Tile sizes can be computed automatically in some cases, using
      the `--cinm-infer-tile-sizes` pass. This requires information about the
      hardware backend in the form of an accelerator attribute on an enclosing
      `cinm.compute` op. The accelerator must implement `computeTilingFactors`
      for this to work.
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
            /*defaultImplementation=*/ "">,

        InterfaceMethod<
            /*desc=*/ [{
              Populate dimSizes with the sizes of the tileable dimensions, in
              the same order as the tiling factors passed to convertToTiledOps.
              Dynamic dimension sizes are represented as ShapedType::kDynamic.
            }],
            /*retTy=*/ "void",
            /*methodName=*/ "getTilableDimSizes",
            /*args=*/ (ins "SmallVectorImpl<int64_t>&":$dimSizes),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ "">];
}

#endif
