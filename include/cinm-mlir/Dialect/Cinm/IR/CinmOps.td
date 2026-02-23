//===- Ops.td - Cinm dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_OPS
#define CINM_OPS


include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmAttributes.td"
include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgEnums.td"

def Cinm_ElementwiseOp
    : Cinm_Op<"op.elementwise", [
        Pure,
        DeclareOpInterfaceMethods<CinmTilingInterface>,
        AttrSizedOperandSegments
    ]> {

    let summary = "";
    let description = [{
        TODO
    }];

    let arguments = (ins
        Cinm_ElementwiseKindAttr:$kind,
        AnyShaped:$lhs,
        Optional<AnyShaped>:$rhs,
        Optional<AnyMemRef>:$out
    );

    let results = (outs Optional<AnyRankedTensor>:$result);

    let extraClassDeclaration = [{
    }];

    let hasCustomAssemblyFormat = true;

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins "cinm::ElementwiseKind":$kind, "Value":$a, "Value":$b, "Value":$out)>,
      OpBuilder<(ins "cinm::ElementwiseKind":$kind, "Value":$a, "Value":$b),
        "build($_builder, $_state, kind, a, b, Value{});">,
      OpBuilder<(ins "cinm::ElementwiseKind":$kind, "Value":$a),
        "build($_builder, $_state, kind, a, Value{}, Value{});">
    ];
}

class Cinm_GemmlikeOp<string name, string cppClassName> : Cinm_Op<name,
  [
    Pure,
    InferTensorTypeAdaptor,
    DeclareOpInterfaceMethods<CinmTilingInterface>,
    // OptionalTypesMatchWith<
    //   "bias should have compatible type", "result", "bias", "$_self">,
    AttrSizedOperandSegments
  ]> {

  let arguments = (ins
      AnyShaped:$lhs,
      AnyShaped:$rhs,
      Optional<AnyShaped>:$bias,
      Optional<AnyMemRef>:$out
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let assemblyFormat =
    "$lhs `,` $rhs (`plus` $bias^)? (`into` $out^)? attr-dict `:` type($lhs) `,` type($rhs) (`plus` type($bias)^)? (`into` type($out)^):(`->` type($result))?";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$left, "Value":$right, "Value":$bias, "Value":$out),
      "buildGemmLikeOp<" # cppClassName # ">($_builder, $_state, left, right, bias, out);">,
    OpBuilder<(ins "Value":$left, "Value":$right, "Value":$bias),
      "build($_builder, $_state, left, right, bias, Value{});">,
    OpBuilder<(ins "Value":$left, "Value":$right),
      "build($_builder, $_state, left, right, Value{}, Value{});">
  ];
}

def Cinm_GemmOp : Cinm_GemmlikeOp<"op.gemm", "GemmOp"> {

  let summary = "Generalized Matrix Matrix Multiplication operation";

}

def Cinm_GemvOp : Cinm_GemmlikeOp<"op.gemv", "GemvOp"> {
  let summary = "Generalized Matrix Vector Multiplication";
}

def Cinm_BatchGemmOp : Cinm_GemmlikeOp<"op.batch_gemm", "BatchGemmOp"> {
  let summary = "Batched matrix multiplication";
  let description = [{}];
}

def Cinm_BatchGemvOp : Cinm_GemmlikeOp<"op.batch_gemv", "BatchGemvOp"> {
  let summary = "Batched matrix vector multiplication";
  let description = [{}];
}


/// we infer the shape because verifying correctness of a
/// hardcoded return type is as hard as inferring it.
def Cinm_TransposeOp : Cinm_Op<"op.transpose", [InferShapedTypeOpAdaptor, Pure]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    AnyShaped:$input1,
    AnyShaped:$perms // TODO
  );

  let results = (
    outs AnyShaped:$output
  );

  let extraClassDeclaration = [{
    LogicalResult getConstantPerms(llvm::SmallVector<int64_t> &perms);
  }];
}


def Cinm_ScanOp : Cinm_Op<"op.scan", [SameOperandsAndResultType, Pure]> {
  let summary = "Scan the tensor and return a tensor with the same shape.";

  let description = [{

  }];

  let arguments = (ins
    Cinm_ScanMethodAttr:$method,
    AnyShaped:$input
  );

  let results = (
    outs
    AnyShaped:$result
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$method `(` $input `)` attr-dict `:` type($input)";
}

def Cinm_ReduceOp : Cinm_Op<"op.reduce",
    [Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Reduce the tensor according to a given binary operator.";

  let description = [{
    Add a dimension using the `dimensions` attribute. For now, only 1 reduction
    dimension is allowed. Can be negative to index from the end. Default value
    is -1 (last dim). 0-ranked result gets converted to scalar value. Example:

    ```
    %r = cinm.compute attributes { workgroupShape = array<i64: 1,8,16> } -> tensor<288xf32> {
        // Simple vector to scalar: tensor<100xf32> -> f32
        %reduced_vec = cinm.op.reduce min (%vec) : tensor<100xf32> -> f32
        // Default: reduction dimension = -1 -> last dim -> on a 2D tensor, dim = 1
        %max_along_dim1_a = cinm.op.reduce max (%x) : tensor<288x512xf32> -> tensor<288xf32>
        // Equivalent
        %max_along_dim1_b = cinm.op.reduce max (%x) { dimensions = array<i64: 1> } : tensor<288x512xf32> -> tensor<288xf32>
        // Different dimension yields different result type tensor<512xf32>
        %add_along_dim0 = cinm.op.reduce add (%x) { dimensions = array<i64: 0> } : tensor<288x512xf32> -> tensor<512xf32>
        cinm.yield %max_along_dim1_a : tensor<288xf32>
    }
    ```
  }];

  let arguments = (ins
    Cinm_ReduceMethodAttr:$method,
    AnyShaped:$input,
    DefaultValuedAttr<DenseI64ArrayAttr, "{-1}">:$dimensions
  );

  let results = (
    outs
    AnyType:$result
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$method `(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

def Cinm_TopKOp : Cinm_Op<"op.topK", [InferShapedTypeOpAdaptor, Pure]> {
  let summary = "Top K elements";

  let description = [{

  }];

  let arguments = (ins
    // todo what is enum parameter? it's an attribute though
    // two tensor parameters to search, TODO must have same shape
    AnyShaped:$input,
    I64:$k // size of result tensor
  );

  let results = (
    outs
    // these have dynamic sizes and rank 1
    AnyShaped:$resultValues,
    TensorOf<[Builtin_Index]>:$resultIndices
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$k `(` $input `)` attr-dict `:` type($input) `->` type($resultValues) `,` type($resultIndices)";
}

def Cinm_SimSearchOp : Cinm_Op<"op.simSearch",
  [InferTensorTypeAdaptor, Pure,
   TypesMatchWith<"Operands have same type", "left", "right", "$_self">]> {

  let summary = "Similarity search between two tensors";

  let description = [{

    Similarity search between tensors or the same shape. Two similarity metrics
    are supported: cos and dot.

        %sim1, %sim1i = cinm.op.simSearch cos 4 (%scan, %scan2) : tensor<6x6xi32>

    The type trailing the operator is the type of both operands.
    The results are of dynamic shape, the first are values, the
    second are indices of the values. Here they are of type `tensor<?xi32>` and `tensor<?xindex>`

  }];

  let arguments = (ins
    // two tensor parameters to search, TODO must have same shape
    Cinm_SimilarityMetricAttr:$metric,
    AnyShaped:$left,
    AnyShaped:$right,
    Builtin_IntegerAttr:$k // size of result tensors (maybe max size)
  );

  let results = (
    outs
    // these have dynamic sizes and rank 1
    AnyShaped:$resultValues,
    AnyShaped:$resultIndices
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$metric $k `(` $left `,` $right `)` attr-dict `:` type($left)";
}

def Cinm_MergePartialOp : Cinm_Op<"op.mergePartial", [Pure]> {
  let summary = "Merge partial results of a sim search (?)";

  let description = [{
    %sim1, %sim1i = cinm.op.simSearch cos 4 (%scan, %scan2) : tensor<6x6xi32>
    %sim2, %sim2i = cinm.op.simSearch dot 4 (%scan, %scan2) : tensor<6x6xi32>
  );

  let results = (
    outs
    AnyShaped:$result
  );

  let extraClassDeclaration = [{
  }];
}

def Cinm_PopCountOp : Cinm_Op<"op.popCount", [Pure]> {
  let summary = "Pop count on a vector of i1";

  let description = [{

  }];

  let arguments = (ins
    TensorOf<[I1]>:$left
  );

  let results = (
    outs
    I64:$result
  );

  let extraClassDeclaration = [{
  }];
}



def Cinm_ComputeOp : Cinm_Base_Op<"compute", [
  AffineScope, AutomaticAllocationScope, IsolatedFromAbove
]> {
  let summary = "TODO";
  let description = [{

    %rbuf = cinm.compute attributes { ... } -> tensor<...> {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %arg0, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
  }];

  let arguments = (ins
      Variadic<AnyType>:$operands,
      // Variadic<AnyType>:$outputs,

      DefaultValuedAttr<DenseI64ArrayAttr, "{2, 32, 16}">:$workgroupShape,
      DefaultValuedAttr<DenseI64ArrayAttr, "{0, 65536, 0}">:$bufferSizesInBytes,
      OptionalAttr<DenseI64ArrayAttr>:$tileSizes
  );

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    cnm::WorkgroupType getCnmWorkgroupType() {
      return cnm::WorkgroupType::get(getContext(), getWorkgroupShape());
    }
    long getWorkgroupSize() {
      auto wgShape = getWorkgroupShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
    auto zipArgsWithOperands() {
      return llvm::zip(getRegion().getArguments(), getOperands());
    }
    auto zipArgsWithOpOperands() {
      return llvm::zip(getRegion().getArguments(), (*this)->getOpOperands());
    }
    auto getBodyArguments() {
      return getRegion().getArguments();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "TypeRange":$results), [{
      auto wg   = $_builder.getDenseI64ArrayAttr({2, 32, 16});
      auto bufs = $_builder.getDenseI64ArrayAttr({0, 65536, 0});

      $_state.addOperands(operands);
      $_state.addTypes(results);
      $_state.addAttribute("workgroupShape", wg);
      $_state.addAttribute("bufferSizesInBytes", bufs);
      // Do NOT set tileSizes here (absent means "use heuristic").

      ::mlir::Region *region = $_state.addRegion();
      auto& block = region->emplaceBlock(); // satisfy SizedRegion<1>
      for (auto opnd : operands)
        block.addArgument(opnd.getType(), opnd.getLoc());
    }]>
  ];

 let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//
def Cinm_SelectOp : Cinm_Op<"select", [Pure]> {
  let summary = "Select one of multiple child-regions depending on the cost-model estimates";
  let description = [{}];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let assemblyFormat = "attr-dict `from` $regions (`->` type($results)^)?";
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//
def Cinm_YieldOp : Cinm_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "Yield the result of a cinm.compute / cinm.select.";
  let description = [{
    Terminator for both `cinm.compute` and `cinm.select`.
    The operand types must match the parent op's result types exactly.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Cinm_ActivateOp : Cinm_Op<"op.activate", [
    Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<CinmTilingInterface>
  ]> {

  let summary = "Elementwise activation (ReLU/Sigmoid/Tanh/GELU)";
  let description = [{
    Applies the chosen activation function elementwise to the input tensor.

    Examples:
    ```
    %y = cinm.op.activate relu (%x) : tensor<1x64x56x56xf32>
    %z = cinm.op.activate sigmoid (%y) : tensor<1x64x56x56xf32>
    ```
  }];

  let arguments = (ins
    Cinm_ActivationKindAttr:$kind,
    AnyShaped:$input,
    Optional<AnyMemRef>:$out
  );
  let results = (outs Optional<AnyRankedTensor>:$result);

  let skipDefaultBuilders = 1;
  let builders=[
    OpBuilder<(ins "ActivationKind":$kind, "Value":$input, "Value":$output)>,
    OpBuilder<(ins "ActivationKind":$kind, "Value":$input),
        "build($_builder, $_state, kind, input, Value{});">,
  ];

  let hasCustomAssemblyFormat = true;
}

def Cinm_QuantizeOp : Cinm_Op<"op.quantize", [Pure]> {
  let summary = "Quantize tensor (float -> integer) with per-tensor params";
  let description = [{
    Per-tensor affine quantization of a tensor:
      out[i...] = clamp(round(src[i...] / scale) + zeroPoint, qmin, qmax)

    Contracts (enforced later via verifier/lowering, not here):
      - element_type(src) ∈ {f16, bf16, f32, f64}
      - element_type(result) is integer (e.g. i8/i16/i32 or signless of those)
      - ranks & shapes of src and result match
      - `scale` is positive, `zeroPoint` fits in result element type
      - `narrowRange` uses (qmin+1 .. qmax) instead of (qmin .. qmax)
      - `axis` reserved for future per-channel support (ignored for now)

    Example:
      %q = cinm.op.quantize %x
           { scale = 0.03125 : f32, zeroPoint = 0 : i64,
             rounding = #cinm.rounding<nearest> }
           : tensor<1024x1024xf32> -> tensor<1024x1024xi8>
  }];

  let arguments = (ins
    AnyShaped:$src,
    Optional<AnyMemRef>:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<Cinm_RoundingMode, "::mlir::cinm::RoundingMode::Nearest">:$rounding,
    DefaultValuedAttr<BoolAttr, "false">:$narrowRange
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasCustomAssemblyFormat = true;
}

def Cinm_DequantizeOp : Cinm_Op<"op.dequantize", [Pure]> {
  let summary = "Dequantize tensor (integer -> float) with per-tensor params";
  let description = [{
    Per-tensor affine dequantization of a tensor:
      out[i...] = (src[i...] - zeroPoint) * scale

    Contracts (to be verified later):
      - element_type(src) is integer (e.g. i8/i16/i32 or signless of those)
      - element_type(result) ∈ {f16, bf16, f32, f64}
      - ranks & shapes match
      - `axis` reserved for future per-channel (ignored for now)

    Example:
      %x = cinm.op.dequantize %q
           { scale = 0.03125 : f32, zeroPoint = 0 : i64 }
           : tensor<1024x1024xi8> -> tensor<1024x1024xf32>
  }];

  let arguments = (ins
    AnyShaped:$src,
    Optional<AnyMemRef>:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasCustomAssemblyFormat = true;
}

#endif
