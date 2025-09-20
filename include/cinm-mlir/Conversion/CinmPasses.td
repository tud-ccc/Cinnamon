//===- Passes.td - Cinm dialect passes ---------------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef CINM_CONVERSION_PASSES
#define CINM_CONVERSION_PASSES


include "mlir/Pass/PassBase.td"

def ConvertTiledCinmToCnm : Pass<"convert-cinm-to-cnm", operation="func::FuncOp"> {
    let summary = "Convert CINM dialect to CNM dialect (may require --cinm-tiling before)";
    let constructor = "mlir::cinm::createConvertTiledCinmToCnmPass()";
    let dependentDialects = ["cnm::CnmDialect", "func::FuncDialect", 
        "arith::ArithDialect", "tensor::TensorDialect"];
}

def ConvertTiledCinmToCim : Pass<"convert-cinm-to-cim", operation="func::FuncOp"> {
    let summary = "Convert CINM dialect to CIM dialect (may require --cinm-tiling before)";
    let constructor = "mlir::cinm::createConvertTiledCinmToCimPass()";
    let dependentDialects = ["cim::CimDialect", "func::FuncDialect", "tensor::TensorDialect"];
}

def ConvertLinalgToCinm
  : Pass<"convert-linalg-to-cinm", operation="func::FuncOp"> {
  let summary = "Convert Linalg matmul/matvec to CINM gemm/gemv (tensor or memref)";
  let description = [{
    Rewrites Linalg matrix-multiply kernels to CINM ops.

    Handled patterns (both named ops and, optionally, their canonical
    linalg.generic forms):

      * `linalg.matmul`
          - tensor operands/results → `cinm.op.gemm`
            (fuse bias if the init/outs or an immediate add matches MxN)
          - memref operands/results → `cinm.gemm_memref` (writes into outs)

      * `linalg.matvec`
          - tensor operands/results → `cinm.op.gemv`
            (fuse bias if the init/outs or an immediate add matches M)
          - memref operands/results → `cinm.gemv_memref` (writes into outs)

    Bias fusion (optional, enabled by default):
      - If the matmul/matvec uses an `outs`/init tensor/memref produced from a
        value `B` with matching shape, or the result is consumed by a single
        elementwise add with a compatible `B`, the pass attaches `B` as the
        optional `plus` operand on `cinm.op.gemm` / `cinm.op.gemv`. For memref
        forms, the `out` buffer already carries the initial contents.
  }];

  let constructor = "mlir::cinm::createConvertLinalgToCinmPass()";

  let dependentDialects = [
    "cinm::CinmDialect",
    "linalg::LinalgDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "memref::MemRefDialect"
  ];

  let options = [
    Option<"fuseBias", "fuse-bias", "bool", "true",
           "Try to detect/fuse +bias epilogues into CINM ops">,
    Option<"matchGeneric", "match-generic", "bool", "true",
           "Also match linalg.generic forms of matmul/matvec">,
    Option<"foldTransposeOnRhs", "fold-transpose-on-rhs", "bool", "true",
           "Accept/peel a simple linalg.transpose on RHS when mapping to GEMM">
  ];
}

def Im2ColToMatmul
  : Pass<"im2col-to-matmul", operation="func::FuncOp"> {
  let summary = "Turn supported linalg.conv ops into matmul shapes via im2col";
  let constructor = "mlir::cinm::createIm2ColToMatmulPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "arith::ArithDialect"
  ];
}


#endif // CINM_CONVERSION_PASSES
