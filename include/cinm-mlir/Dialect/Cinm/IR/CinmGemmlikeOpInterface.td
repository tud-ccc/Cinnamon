//===- CinmGemmlikeOpInterface.td ----------------*- tablegen -*-===//
//
// Defines the GemmlikeOpInterface for CINM GEMM-like operations.
//
//===--------------------------------------------------------------===//

#ifndef CINM_GEMMLIKE_OP_INTERFACE
#define CINM_GEMMLIKE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def CinmGemmlikeOpInterface : OpInterface<"GemmlikeOpInterface"> {
    let cppNamespace = "::mlir::cinm";

    let description = [{
      Interface shared by all CINM GEMM-like operations (`cinm.op.gemm`,
      `cinm.op.gemv`, `cinm.op.batch_gemm`, `cinm.op.batch_gemv`).
      Provides uniform access to the common lhs / rhs / bias / out / result
      structure.

      The op can operate in two modes:
      - **Tensor mode**: operands are tensors; the op returns a result tensor.
      - **Memref mode**: operands are memrefs; the op accumulates into `out`
        in place and has no result.

      Use `isTensorVariant()` to distinguish between the two.
    }];

    let methods = [
        InterfaceMethod<
            "Returns the left-hand-side operand.",
            "::mlir::Value",
            "getLhs",
            (ins),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ [{ return $_op.getLhs(); }]>,

        InterfaceMethod<
            "Returns the right-hand-side operand.",
            "::mlir::Value",
            "getRhs",
            (ins),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ [{ return $_op.getRhs(); }]>,

        InterfaceMethod<
            "Returns the optional bias operand, or a null Value if absent.",
            "::mlir::Value",
            "getBias",
            (ins),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ [{ return $_op.getBias(); }]>,

        InterfaceMethod<
            "Returns the optional output-buffer operand, or a null Value if absent.",
            "::mlir::Value",
            "getOut",
            (ins),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ [{ return $_op.getOut(); }]>,

        InterfaceMethod<
            "Returns the result tensor, or a null Value when operating in-place on a memref.",
            "::mlir::Value",
            "getGemmResult",
            (ins),
            /*methodBody=*/ "",
            /*defaultImplementation=*/ [{ return $_op.getResult(); }]>,
    ];

    let extraClassDeclaration = [{
        /// True when the op operates on tensors (produces a result).
        /// False when it accumulates in-place into a memref output buffer.
        bool isTensorVariant() { return static_cast<bool>(getGemmResult()); }
    }];
}

#endif
