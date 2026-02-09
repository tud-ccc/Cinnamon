//===- Ops.td - Cim dialect ops ----------------------*- tablegen -*-===//
#ifndef CIM_OPS
#define CIM_OPS

include "cinm-mlir/Dialect/Cim/IR/CimBase.td"
include "cinm-mlir/Dialect/Cim/IR/CimTypes.td"
include "cinm-mlir/Dialect/Cim/IR/CimAttributes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/CommonAttrConstraints.td"

//===----------------------------------------------------------------------===//
// Common type constraints (memref-only API)
//===----------------------------------------------------------------------===//

// Inputs must be memrefs, or a CIM future whose payload is a memref.
def Cim_OpInputMemLike :
  TypeConstraint<Or<[
    CPred<[{ llvm::isa<MemRefType>($_self) }]>,
    CPred<[{ llvm::isa<cim::FutureType>($_self) }]>
  ]>> {}

// Barrier resolves a future into its memref payload.
def Cim_MemRef :
  TypeConstraint<CPred<[{ llvm::isa<MemRefType>($_self) }]>> {}

//===----------------------------------------------------------------------===//
// Administrative ops (unchanged)
//===----------------------------------------------------------------------===//

def AcquireDeviceOp : Cim_Op<"acquire_device",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Acquire a cim device, returns id.";
  let arguments = (ins
    DefaultValuedAttr<I32Attr, "1">:$availableCrossbarCount,
    DefaultValuedAttr<BoolAttr, "false">:$isFullyScheduled
  );
  let results = (outs DeviceIdType:$deviceId);
  let assemblyFormat = "attr-dict `->` qualified(type($deviceId))";
}

def ReleaseDeviceOp : Cim_Op<"release_device", []> {
  let arguments = (ins DeviceIdType:$deviceId);
  let results = (outs);
  let assemblyFormat = "$deviceId attr-dict `:` qualified(type($deviceId))";
}

def AcquireCrossbarOp : Cim_Op<"acquire_crossbar",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Acquire a cim crossbar, returns id.";
  let arguments = (ins
    DeviceIdType:$deviceId,
    OptionalAttr<I64Attr>:$height,
    OptionalAttr<I64Attr>:$width
  );
  let results = (outs CrossbarIdType:$crossbarId);
  let assemblyFormat = "$deviceId attr-dict `:` qualified(type($deviceId)) `->` qualified(type($crossbarId))";
}

def ReleaseCrossbarOp : Cim_Op<"release_crossbar", []> {
  let arguments = (ins CrossbarIdType:$crossbarId);
  let results = (outs);
  let assemblyFormat = "$crossbarId attr-dict `:` qualified(type($crossbarId))";
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def BarrierOp : Cim_Op<"barrier", []> {
  let summary = "Wait for dependent computations to finish; returns memref payload.";
  let arguments = (ins FutureType:$value);
  let results = (outs Cim_MemRef:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Compute ops (memref-only)
//===----------------------------------------------------------------------===//

def GemmOp : Cim_Op<"op.gemm", []> {
  let summary = "Matrix×matrix multiply; memref-only API.";
  let arguments = (ins
    CrossbarIdType:$crossbarId,
    AnyStridedMemRefOfRank<2>:$lhs,
    AnyStridedMemRefOfRank<2>:$rhs
  );
  // Future payload must be a memref<…x…>
  let results = (outs FutureType:$result);
  let assemblyFormat =
    "$crossbarId `,` $lhs `,` $rhs attr-dict `:` "
    "qualified(type($crossbarId)) `,` type($lhs) `,` type($rhs) "
    "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def GemvOp : Cim_Op<"op.gemv", []> {
  let summary = "Matrix×vector multiply; memref-only API.";
  let arguments = (ins
    CrossbarIdType:$crossbarId,
    AnyStridedMemRefOfRank<2>:$lhs,
    AnyStridedMemRefOfRank<1>:$rhs
  );
  // Future payload must be a memref<…>
  let results = (outs FutureType:$result);
  let assemblyFormat =
    "$crossbarId `,` $lhs `,` $rhs attr-dict `:` "
    "qualified(type($crossbarId)) `,` type($lhs) `,` type($rhs) "
    "`->` qualified(type($result))";
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Elementwise add (memref-only)
//===----------------------------------------------------------------------===//

def AddOp : Cim_Op<"op.add", []> {
  let summary = "Elementwise add; memref-only API.";
  let description = [{
    Computes elementwise addition on two memrefs of the same shape and element
    type. Returns a !cim.future whose payload is a memref with the same shape
    and element type as the inputs.

    Contract (checked in verifier):
      - `lhs` and `rhs` are memrefs (any rank) with identical shape & element type.
      - Result future's payload type is that same memref type.
  }];

  let arguments = (ins
    CrossbarIdType:$crossbarId,
    Cim_MemRef:$lhs,
    Cim_MemRef:$rhs
  );

  let results = (outs FutureType:$result);

  let assemblyFormat =
    "$crossbarId `,` $lhs `,` $rhs attr-dict `:` "
    "qualified(type($crossbarId)) `,` type($lhs) `,` type($rhs) "
    "`->` qualified(type($result))";

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Quantization (memref-only; returns !cim.future)
//===----------------------------------------------------------------------===//
def QuantizeOp : Cim_Op<"op.quantize", []> {
  let summary = "Quantize memref (float -> integer); returns !cim.future<memref<...>>";
  let description = [{
    Per-tensor affine quantization of a memref, producing a new memref on the
    device. The operation returns a `!cim.future` whose payload is that result
    memref. Attribute names/semantics match CINM.
  }];

  let arguments = (ins
    CrossbarIdType:$crossbarId,
    Cim_MemRef:$src,

    F32Attr:$scale,
    I64Attr:$zeroPoint,

    OptionalAttr<I64Attr>:$axis,

    DefaultValuedAttr<Cim_RoundingMode,
                      "::mlir::cim::RoundingMode::Nearest">:$rounding,
    DefaultValuedAttr<BoolAttr, "false">:$narrowRange
  );

  let results = (outs FutureType:$result);

  let assemblyFormat =
    "$crossbarId `,` $src attr-dict `:` "
    "qualified(type($crossbarId)) `,` type($src) "
    "`->` qualified(type($result))";

  let hasVerifier = 1;
}

def DequantizeOp : Cim_Op<"op.dequantize", []> {
  let summary = "Dequantize memref (integer -> float); returns !cim.future<memref<...>>";
  let description = [{
    Per-tensor affine dequantization matching CINM’s attributes and semantics.
    Takes an integer-typed memref and returns a future whose payload is a float
    memref with the same shape.
  }];

  let arguments = (ins
    CrossbarIdType:$crossbarId,
    Cim_MemRef:$src,

    F32Attr:$scale,
    I64Attr:$zeroPoint,

    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs FutureType:$result);

  let assemblyFormat =
    "$crossbarId `,` $src attr-dict `:` "
    "qualified(type($crossbarId)) `,` type($src) "
    "`->` qualified(type($result))";

  let hasVerifier = 1;
}

def Cim_ActivateOp : Cim_Op<"op.activate", []> {
  let summary = "Elementwise activation (future form; memref in, future out)";
  let description = [{
    Applies an activation elementwise to a memref input and returns a
    !cim.future of the output memref. Materialize via `cim.barrier`.

    Example:
      %f = cim.op.activate relu (%x)
           : memref<?x?xf32> -> !cim.future<memref<?x?xf32>>
      %y = cim.barrier %f
           : !cim.future<memref<?x?xf32>> -> memref<?x?xf32>
  }];

  let arguments = (ins
    Cim_ActivationKindAttr:$kind,
    AnyMemRef:$input
  );

  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$kind `(` $input `)` attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}


#endif // CIM_OPS
