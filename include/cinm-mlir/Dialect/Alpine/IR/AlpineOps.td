//===- Ops.td - Alpine dialect ops --------------------------*- tablegen -*-===//
//
// Core ops for the Alpine crossbar device.
//
//===----------------------------------------------------------------------===//

#ifndef ALPINE_OPS_TD
#define ALPINE_OPS_TD

include "cinm-mlir/Dialect/Alpine/IR/AlpineBase.td"
include "cinm-mlir/Dialect/Alpine/IR/AlpineTypes.td"
include "cinm-mlir/Dialect/Alpine/IR/AlpineAttributes.td"

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

//------------------------------------------------------------------------------
// alpine.alloc_tile  (returns a tile id, e.g. i32)
//------------------------------------------------------------------------------
def Alpine_AllocTileOp : Alpine_Op<"alloc_tile",
    [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Allocate/configure a tile and return its tile id (host handle)";
  let description = [{
    Returns a host-side tile identifier and optionally configures the
    crossbar dimensions in the backend runtime.

    Attributes (optional):
    - height: i64 — emulated crossbar height
    - width:  i64 — emulated crossbar width

    Example:
    ```
    %tile = alpine.alloc_tile {height = 128, width = 128} -> i32
    ```
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$height,
    OptionalAttr<I64Attr>:$width
  );

  let results = (outs SignlessIntegerLike:$tileId);

  let assemblyFormat = "attr-dict `->` qualified(type($tileId))";
}


//------------------------------------------------------------------------------
// alpine.write_weights
//------------------------------------------------------------------------------
def Alpine_WriteWeightsOp : Alpine_Op<"write_weights"> {
  let summary = "Write a weight matrix into a tile's crossbar";
  let description = [{
    Uploads a K×N weight matrix to the crossbar of the given tile. The element
    type of `W` determines precision on the device. `W` must be a contiguous
    memref of rank 2 and fit into the crossbar.

    Example:
    ```
    // B: memref<KxNxf32>, tileId: i32
    alpine.write_weights %tileId, %B : i32, memref<?x?xf32>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$tileId,
    AnyStridedMemRefOfRank<2>:$W
  );

  let assemblyFormat =
    "$tileId `,` $W attr-dict `:` qualified(type($tileId)) `,` type($W)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_write_weights"; }
  }];
  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// alpine.enqueue_vec
//------------------------------------------------------------------------------
def Alpine_EnqueueVecOp : Alpine_Op<"enqueue_vec"> {
  let summary = "Enqueue an input vector to the tile's input queue";
  let description = [{
    Pushes a length-K input vector into the given tile's input queue. The
    device will consume vectors when `alpine.process` is issued.

    Example:
    ```
    alpine.enqueue_vec %tileId, %x : i32, memref<?xf32>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$tileId,
    AnyStridedMemRefOfRank<1>:$x
  );

  let assemblyFormat =
    "$tileId `,` $x attr-dict `:` qualified(type($tileId)) `,` type($x)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_enqueue_vec"; }
  }];
  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// alpine.process (synchronous host-side trigger)
//------------------------------------------------------------------------------
def Alpine_ProcessOp : Alpine_Op<"process">  {
  let summary = "Process queued input vectors on a tile";
  let description = [{
    Triggers processing on a tile. If `count` is provided, the device processes
    up to that many queued vectors; otherwise it processes the entire queue.

    Example:
    ```
    // process all queued vectors with ReLU activation
    alpine.process %tileId { activation = #alpine<activation relu> } : i32

    // process at most 1 vector (used by canonicalized alpine.mvm)
    alpine.process %tileId { count = 1 } : i32
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$tileId,
    OptionalAttr<StrAttr>:$activation,
    DefaultValuedAttr<BoolAttr, "false">:$accumulate,
    OptionalAttr<I64Attr>:$count
  );

  let assemblyFormat =
    "$tileId attr-dict `:` qualified(type($tileId))";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_process"; }
  }];
  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// alpine.dequeue_vec
//------------------------------------------------------------------------------
def Alpine_DequeueVecOp : Alpine_Op<"dequeue_vec"> {
  let summary = "Dequeue one finished output vector from a tile";
  let description = [{
    Pops one result vector from the tile's output queue into `y`.

    Example:
    ```
    alpine.dequeue_vec %tileId, %y : i32, memref<?xf32>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$tileId,
    AnyStridedMemRefOfRank<1>:$y
  );

  let assemblyFormat =
    "$tileId `,` $y attr-dict `:` qualified(type($tileId)) `,` type($y)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_dequeue_vec"; }
  }];
  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// (Optional) alpine.mvm: single-shot convenience op
// Lowered by canonicalization into enqueue→process(count=1)→dequeue.
//------------------------------------------------------------------------------
def Alpine_MVMOp : Alpine_Op<"mvm"> {
  let summary = "Single vector × crossbar-matrix multiply into an output vector";
  let description = [{
    Convenience op that performs one MVM using the tile's current weights:
      y[1×N] = x[1×K] · W[K×N]
    It writes into `y` (no result SSA value). Canonicalization expands to the
    streaming trio.

    Example:
    ```
    alpine.mvm %tileId, %x, %y : i32, memref<?xf32>, memref<?xf32>
    ```
  }];

  let arguments = (ins
    SignlessIntegerLike:$tileId,
    AnyStridedMemRefOfRank<1>:$x,
    AnyStridedMemRefOfRank<1>:$y
  );

  let assemblyFormat =
    "$tileId `,` $x `,` $y attr-dict `:` qualified(type($tileId)) `,` type($x) `,` type($y)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_mvm"; }
  }];
  let hasVerifier = 1;
}


//------------------------------------------------------------------------------
// alpine.quantize  (f32 -> i8)
//------------------------------------------------------------------------------
def Alpine_QuantizeOp : Alpine_Op<"quantize"> {
  let summary = "Quantize memref (f32 -> i8), per-tensor scale & zero";
  let description = [{
    Quantize with simple per-tensor affine parameters:

      out[i...] = clamp(round(src[i...] / scale) + zero, -128, 127)

    Contracts:
      - element_type(src) == f32, element_type(out) == i8
      - ranks & shapes of src and out match exactly
      - attributes:
          * scale: f32, must be finite and > 0
          * zero:  i32, must be in [-128, 127]
  }];

  let arguments = (ins
    AnyStridedMemRef:$src,
    AnyStridedMemRef:$out,
    F32Attr:$scale,
    I32Attr:$zero
  );

  let results = (outs);

  // Attributes print inside the attr-dict (e.g. {scale = 0.05 : f32, zero = 3 : i32})
  let assemblyFormat =
    "$src `,` $out attr-dict `:` type($src) `,` type($out)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_quantize"; }
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// alpine.dequantize  (i8 -> f32)  + per-tensor {scale, zero}
//------------------------------------------------------------------------------
def Alpine_DequantizeOp : Alpine_Op<"dequantize"> {
  let summary = "Dequantize memref (i8 -> f32), per-tensor scale & zero";
  let description = [{
    Dequantize with simple per-tensor affine parameters:

      out[i...] = (src[i...] - zero) * scale

    Contracts:
      - element_type(src) == i8, element_type(out) == f32
      - ranks & shapes of src and out match exactly
      - attributes:
          * scale: f32, must be finite and > 0
          * zero:  i32, must be in [-128, 127]
  }];

  let arguments = (ins
    AnyStridedMemRef:$src,
    AnyStridedMemRef:$out,
    F32Attr:$scale,
    I32Attr:$zero
  );

  let results = (outs);

  let assemblyFormat =
    "$src `,` $out attr-dict `:` type($src) `,` type($out)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_dequantize"; }
  }];

  let hasVerifier = 1;
}

def Alpine_ReluOp : Alpine_Op<"relu"> {
  let summary = "ReLU activation on memref (f32 -> f32), writes into out";
  let description = [{
    Applies ReLU elementwise with no configuration:

      out[i...] = max(src[i...], 0.0f)

    Contracts (enforced by the verifier / lowering):
      - element_type(src) == f32, element_type(out) == f32
      - ranks & shapes of src and out match exactly
  }];

  let arguments = (ins
    AnyStridedMemRef:$src,
    AnyStridedMemRef:$out
  );

  let results = (outs);

  let assemblyFormat =
    "$src `,` $out attr-dict `:` type($src) `,` type($out)";

  code libraryCallName = [{
    std::string getLibraryCallName() { return "alpine_relu"; }
  }];

  let hasVerifier = 1;
}

#endif // ALPINE_OPS_TD
