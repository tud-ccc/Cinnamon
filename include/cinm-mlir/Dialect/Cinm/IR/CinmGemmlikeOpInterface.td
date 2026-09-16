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

        /// Element type of the values being multiplied. Both operands share
        /// it; only the accumulator may be wider.
        ::mlir::Type getOperandElementType() {
            return ::llvm::cast<::mlir::ShapedType>(getLhs().getType())
                .getElementType();
        }

        /// Element type the products accumulate into. `out` and `bias` are
        /// the accumulator whenever they are present -- the tiling pass
        /// threads the running accumulator through `bias`, so honouring it
        /// keeps a tiled op's result type equal to the untiled op's.
        ::mlir::Type getAccumulatorElementType() {
            ::mlir::Value v = getOut();
            if (!v) v = getBias();
            if (!v) v = getGemmResult();
            if (!v) return getOperandElementType();
            return ::llvm::cast<::mlir::ShapedType>(v.getType())
                .getElementType();
        }

        /// True when the operands are narrower than the accumulator, as in
        /// the i8 x i8 -> i32 of an int8 checkpoint.
        bool isMixedPrecision() {
            return getOperandElementType() != getAccumulatorElementType();
        }
    }];
}

#endif
