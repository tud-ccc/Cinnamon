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
include "cinm-mlir/Dialect/Cinm/IR/CinmPlatformAttrInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/TilingInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmComputeOpInterface.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmGemmlikeOpInterface.td"

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
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
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
        Optional<AnyShaped>:$out
    );

    let results = (outs Optional<AnyRankedTensor>:$result);

    let extraClassDeclaration = [{
      ArrayRef<int64_t> getShape() {
        return getLhs().getType().getShape();
      }
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
    InferTensorTypeAdaptor,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    // OptionalTypesMatchWith<
    //   "bias should have compatible type", "result", "bias", "$_self">,
    AttrSizedOperandSegments,
    CinmGemmlikeOpInterface
  ]> {

  let description = [{
    CINM GEMM-like ops (batch/non-batch gemm/gemv) semantics depend on whether they operate
    on memrefs or tensors.
    
    If the operands are tensors:
    ```mlir
        %C = cinm.op.gemm %A, %B                      // %C = %A * %B
        %C = cinm.op.gemm %A, %B plus %bias           // %C = %A * %B + %bias
    ```
    A tensor out buffer is only used as a bufferization hint (is states that %C may be equivalent to %out):
    ```mlir
        %C = cinm.op.gemm %A, %B into %out            // %C = %A * %B
        %C = cinm.op.gemm %A, %B plus %bias into %out // %C = %A * %B + %bias
    ```
    If the operands are memrefs:
    ```mlir
        cinm.op.gemm %A, %B into %out                 // %out += %A * %B
        cinm.op.gemm %A, %B plus %bias into %out      // %out += %A * %B + %bias
    ```
    The main difference is that the memref variant reads from the out buffer and accumulates into it, while the tensor variant does not. When bufferizing, the output buffer is first initialized
    with the bias (or zeroed out if there is no bias) to preserve the tensor semantics.

    Note that this difference can lead to slightly different bufferized IRs depending on whether you bufferize or tile first.

  }];

  let arguments = (ins
      AnyShaped:$lhs,
      AnyShaped:$rhs,
      Optional<AnyShaped>:$bias,
      Optional<AnyShaped>:$out
  );

  // let hasVerifier = 1; // todo
  let hasFolder = 1;
  let results = (outs Optional<AnyRankedTensor>:$result);
  let assemblyFormat =
    "$lhs `,` $rhs (`plus` $bias^)? (`into` $out^)? attr-dict `:` type($lhs) `,` type($rhs) (`plus` type($bias)^)? (`into` type($out)^)? (`->` type($result)^)?";

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
}

def Cinm_BatchGemvOp : Cinm_GemmlikeOp<"op.batch_gemv", "BatchGemvOp"> {
  let summary = "Batched matrix vector multiplication";
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
    DenseI64ArrayAttr:$permutation
  );

  let results = (
    outs AnyShaped:$output
  );

  let extraClassDeclaration = [{
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
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>]> {
  let summary = "Reduce the tensor according to a given binary operator.";

  let description = [{
    Add a dimension using the `dimensions` attribute. For now, only 1 reduction
    dimension is allowed. Can be negative to index from the end. Default value
    is -1 (last dim). 0-ranked result gets converted to scalar value. Example:

    ```
    %r = cinm.compute -> tensor<288xf32> {
        // Simple vector to scalar: tensor<100xf32> -> f32
        %reduced_vec = cinm.op.reduce min (%vec) : tensor<100xf32> -> f32
        // Default: reduction dimension = -1 -> last dim -> on a 2D tensor, dim = 1
        %max_along_dim1_a = cinm.op.reduce max (%x) : tensor<288x512xf32> -> tensor<288xf32>
        // Equivalent
        %max_along_dim1_b = cinm.op.reduce max (%x) { dimension = 1 } : tensor<288x512xf32> -> tensor<288xf32>
        // Different dimension yields different result type tensor<512xf32>
        %add_along_dim0 = cinm.op.reduce add (%x) { dimension = 0 } : tensor<288x512xf32> -> tensor<512xf32>
        cinm.yield %max_along_dim1_a : tensor<288xf32>
    }
    ```
  }];

  let arguments = (ins
    Cinm_ReduceMethodAttr:$method,
    AnyShaped:$input,
    // This attribute is internal and to pass information
    // to the result type inference, in case the result should
    // not be implicitly rank-reduced (scalarized).
    DefaultValuedAttr<BoolAttr, "{true}">:$rankReduce,
    DefaultValuedAttr<I64Attr, "{-1}">:$dimension
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultTy, "cinm::ReduceMethod":$method,  "Value":$input, CArg<"int64_t", "-1">:$dimension)>
    ];

  let results = (
    outs
    AnyType:$result
  );

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
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


def AcceleratorOp : Cinm_Op<"accelerator", [OpAsmOpInterface]> {
  let description = [{
    TileFirst accelerator declaration. This op must be declared at the top of
    a block op.
  }];

  let arguments = (ins CinmAcceleratorAttrInterface:$accelerator);
  let results = (outs AcceleratorRefType:$reference);

  let assemblyFormat = "$accelerator attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void getAsmResultNames(OpAsmSetValueNameFn setName) {
      auto accName = getAccelerator().getPlatform().getName();
      setName(getReference(), accName);
    }
  }];

}

def Cinm_ComputeOp : Cinm_Base_Op<"compute", [
  AffineScope, AutomaticAllocationScope, RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getRegionInvocationBounds", "getSuccessorRegions", "getSuccessorInputs"]>,
  CinmComputeOpInterface
]> {
  let summary = "Delimits a block of IR that is to be offloaded";
  let description = [{
    This operation declares either a platform or an accelerator on
    which the computation is to be done. If none of those are provided,
    CINM's DSE will first assign a platform. The full lowering flow
    requires concretizing a platform into a compatible accelerator,
    which can be used for further lowerings.  

    An accelerator is required for instance to run --cinm-infer-tile-sizes
    or --convert-cinm-to-cnm.

    This operation has no parameters, and is not IsolatedFromAbove.
    Can be turned automatically into an IsolatedFromAbove compute op
    with --cinm-isolate-compute-blocks.

    ```
    %rbuf = cinm.compute -> tensor<...> attributes { ... } {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %arg0, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
    ```
  }];

  let arguments = (ins
      OptionalAttr<CinmPlatformAttrInterface>:$platform,
      OptionalAttr<CinmAcceleratorAttrInterface>:$accelerator
  );

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$results), [{
      $_state.addTypes(results);
      ::mlir::Region *region = $_state.addRegion();
      auto& block = region->emplaceBlock(); // satisfy SizedRegion<1>
    }]>
  ];

 let hasCustomAssemblyFormat = 1;
 let hasCanonicalizer = 1;
 let hasVerifier = 1;
}


def Cinm_ComputeBlockOp : Cinm_Base_Op<"compute_block", [
  AffineScope, AutomaticAllocationScope, IsolatedFromAbove, RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getRegionInvocationBounds", "getSuccessorRegions", "getEntrySuccessorOperands", "getSuccessorInputs"]>,
  CinmComputeOpInterface
]> {
  let summary = "Delimits a region of code for offloading - IsolatedFromAbove";
  let description = [{
    Delimits a region of code for offloading. See `cinm.compute` as well. This
    version is isolated from above to parallelize passes that operate 
    only on the nested IR. 

    ```
    %rbuf = cinm.compute_block (%a = %v : tensor<...>) -> tensor<...>  attributes { ... } {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %a, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }

    %rbuf = cinm.compute_block on platform #upmem.array<4x64> (%a = %v : tensor<...>) -> tensor<...> {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %a, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
    ```
  }];

  let arguments = (ins
      Variadic<AnyType>:$operands,
      OptionalAttr<CinmPlatformAttrInterface>:$platform,
      OptionalAttr<CinmAcceleratorAttrInterface>:$accelerator
  );

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    auto zipArgsWithOperands() {
      return llvm::zip(getRegion().getArguments(), getOperands());
    }
    auto zipArgsWithOpOperands() {
      return llvm::zip(getRegion().getArguments(), (*this)->getOpOperands());
    }
    auto zipResultsWithYieldOperands() {
      auto term = getBody().front().getTerminator();
      return llvm::zip(getResults(), term->getOpOperands());
    }
    auto getBodyArguments() {
      return getRegion().getArguments();
    }

  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "TypeRange":$results), [{

      $_state.addOperands(operands);
      $_state.addTypes(results);
      ::mlir::Region *region = $_state.addRegion();
      auto& block = region->emplaceBlock(); // satisfy SizedRegion<1>
      for (auto opnd : operands)
        block.addArgument(opnd.getType(), opnd.getLoc());
    }]>
  ];

 let hasCustomAssemblyFormat = 1;
 let hasVerifier = 1;
 let hasCanonicalizer = 1;
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
  let summary = "Yield the result of a cinm.compute_block / cinm.select.";
  let description = [{
    Terminator for both `cinm.compute_block` and `cinm.select`.
    The operand types must match the parent op's result types exactly.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
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
    Optional<AnyShaped>:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<Cinm_RoundingMode, "::mlir::cinm::RoundingMode::Nearest">:$rounding,
    DefaultValuedAttr<BoolAttr, "false">:$narrowRange
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  // let hasCustomAssemblyFormat = true;
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
    Optional<AnyShaped>:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  // let hasCustomAssemblyFormat = true;
}

#endif
