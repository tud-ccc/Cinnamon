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


// Define the attribute enums matching elementwise op kind (e.g., add).
def ElementwiseKindAttr : EnumAttr<Cinm_Dialect,
                                   UnaryFn, "elemwise_kind"> {
  // EnumAttr directly inherits the cppNamespace of the UnaryFn enumInfo, which
  // is mlir::linalg in this case. However, we want to override to mlir::cinm.
  let cppNamespace = "mlir::cinm";
}



// Op templates

defvar AnyScalar = AnyType; // todo

// no implicit broadcast
class Cinm_Elementwise_Op<string name, list<Trait> traits = []>
    : Cinm_Op<name, traits # [SameOperandsAndResultType]> {

    let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Cinm_Elementwise_Scalar_Op<string name, list<Trait> traits = []>
    : Cinm_Op<name, traits # [
        SameOperandsAndResultElementType,
        TypesMatchWith<"lhs & result must have the same type", "lhs", "result", "$_self">,
        TypesMatchWith<"rhs must be element type of lhs", "lhs", "rhs", "::llvm::cast<RankedTensorType>($_self).getElementType()">
     ]> {

    let arguments = (ins AnyRankedTensor:$lhs, AnyType:$rhs);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

class Cinm_Bitwise_Op<string name, list<Trait> traits = []>
    : Cinm_Op<name, traits> {
    let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
    let results = (outs AnyRankedTensor:$result);
}

def Cinm_Elementwise_Unary_Op : Cinm_Op<"op.element_wise", [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Generic elementwise unary operation on a tensor";
  let description = [{
    Perform a unary operation on each element of a tensor. Example:

    ```
	%r = cinm.compute attributes { workgroupShape = array<i64: 1,1,16> } -> tensor<512xf32> {
	    %sqrts = cinm.op.element_wise sqrt (%input) : tensor<512xf32>
	    cinm.yield %sqrts : tensor<512xf32>
    ```
  }];

  let arguments = (ins
    ElementwiseKindAttr:$method,
    AnyRankedTensor:$input
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = "$method `(` $input `)` attr-dict `:` type($input)";
}

// Concrete op definitions

class TypeMatchesElementType<string tensorTy, string toInfer>:
TypesMatchWith<"output type matches element type of input", toInfer, tensorTy,
              "::llvm::cast<RankedTensorType>($_self).getElementType()">;

def Cinm_AddOp : Cinm_Elementwise_Op<"op.add", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor addition operation";
  let description = [{}];
}

def Cinm_AddsOp : Cinm_Elementwise_Scalar_Op<"op.adds", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor & scalar addition operation";
  let description = [{}];
}

def Cinm_SubOp : Cinm_Elementwise_Op<"op.sub", [Pure,  DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor subtraction operation";
  let description = [{}];
}

def Cinm_SubsOp : Cinm_Elementwise_Scalar_Op<"op.subs", [Pure,  DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor & scalar subtraction operation";
  let description = [{}];
}

// these two aren't in the paper
def Cinm_MulOp : Cinm_Elementwise_Op<"op.mul", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Hadamard product (element-wise multiplication)";
  let description = [{}];
}

def Cinm_MulsOp : Cinm_Elementwise_Scalar_Op<"op.muls", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor & scalar multiplication operation";
  let description = [{}];
}

def Cinm_DivOp : Cinm_Elementwise_Op<"op.div", [Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor division operation";
  let description = [{}];
}

def Cinm_DivsOp : Cinm_Elementwise_Scalar_Op<"op.divs", [Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor & scalar division operation";
  let description = [{}];
}

def Cinm_GemmOp : Cinm_Op<"op.gemm",
    [Pure, InferTensorTypeAdaptor, DeclareOpInterfaceMethods<CinmTilingInterface>,
     OptionalTypesMatchWith<"bias should have compatible type", "result", "bias", "$_self">]> {
  let summary = "Generalized Matrix Matrix Multiplication operation";
  let description = [{
    TODO
  }];
  let arguments = (ins 2DTensorOf<[AnyType]>:$left,
                       2DTensorOf<[AnyType]>:$right,
                       Optional<2DTensorOf<[AnyType]>>:$bias);
  let results = (outs 2DTensorOf<[AnyType]>:$result);
  let assemblyFormat = "$left `,` $right (`plus` $bias^)? attr-dict `:` `(` type($left) `,` type($right) `)` `->` type($result)";

  let builders=[
    OpBuilder<(ins "Value":$left, "Value":$right),
      "build($_builder, $_state, ValueRange {left, right});">

  ];
}

def Cinm_GemvOp : Cinm_Op<"op.gemv",
  [ Pure,
    DeclareOpInterfaceMethods<CinmTilingInterface>,
    OptionalTypesMatchWith<
      "bias should have compatible type", "result", "bias", "$_self"> ]> {
  let summary = "Generalized Matrix Vector Multiplication";

  let arguments = (ins
    2DTensorOf<[AnyType]>:$left,     
    1DTensorOf<[AnyType]>:$right,  
    Optional<1DTensorOf<[AnyType]>>:$bias 
  );

  let results = (outs 1DTensorOf<[AnyType]>:$result); // M

  let assemblyFormat =
    "$left `,` $right (`plus` $bias^)? attr-dict `:` "
    "`(` type($left) `,` type($right) (`,` type($bias)^)? `)` "
    "`->` type($result)";

  let builders = [
    OpBuilder<(ins "Type":$result, "Value":$left, "Value":$right),
      "build($_builder, $_state, TypeRange{result}, ValueRange{left, right});">
  ];
}

def Cinm_BatchGemmOp : Cinm_Op<"op.batch_gemm",
    [Pure, InferTensorTypeAdaptor, DeclareOpInterfaceMethods<CinmTilingInterface>,
     OptionalTypesMatchWith<"bias should have compatible type", "result", "bias", "$_self">]> {
  let summary = "Batched matrix multiplication";
  let description = [{}];

  let arguments = (ins AnyRankedTensor:$left,
                        AnyRankedTensor:$right,
                        Optional<AnyRankedTensor>:$bias);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$left `,` $right (`plus` $bias^)? attr-dict `:` `(` type($left) `,` type($right) `)` `->` type($result)";

  let builders=[
    OpBuilder<(ins "Value":$left, "Value":$right),
      "build($_builder, $_state, ValueRange{left, right});">
  ];
}

def Cinm_BatchGemvOp : Cinm_Op<"op.batch_gemv",
    [Pure, InferTensorTypeAdaptor, DeclareOpInterfaceMethods<CinmTilingInterface>,
     OptionalTypesMatchWith<"bias should have compatible type", "result", "bias", "$_self">]> {
  let summary = "Batched matrix vector multiplication";
  let description = [{}];

  let arguments = (ins AnyRankedTensor:$left,
                        AnyRankedTensor:$right,
                        Optional<AnyRankedTensor>:$bias);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$left `,` $right (`plus` $bias^)? attr-dict `:` `(` type($left) `,` type($right) (`,` type($bias)^)? `)` `->` type($result)";

  let builders = [
    OpBuilder<(ins "Type":$result, "Value":$left, "Value":$right),
      "build($_builder, $_state, TypeRange{result}, ValueRange{left, right});">
  ];
}


/// we infer the shape because verifying correctness of a
/// hardcoded return type is as hard as inferring it.
def Cinm_TransposeOp : Cinm_Op<"op.transpose", [InferShapedTypeOpAdaptor, Pure]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    AnyRankedTensor:$input1,
    AnyRankedTensor:$perms // TODO
  );

  let results = (
    outs AnyRankedTensor:$output
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
    AnyRankedTensor:$input
  );

  let results = (
    outs
    AnyRankedTensor:$result
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
    AnyNon0RankedTensor:$input,
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
    AnyRankedTensor:$input,
    I64:$k // size of result tensor
  );

  let results = (
    outs
    // these have dynamic sizes and rank 1
    AnyRankedTensor:$resultValues,
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
    AnyRankedTensor:$left,
    AnyRankedTensor:$right,
    Builtin_IntegerAttr:$k // size of result tensors (maybe max size)
  );

  let results = (
    outs
    // these have dynamic sizes and rank 1
    AnyRankedTensor:$resultValues,
    AnyRankedTensor:$resultIndices
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
    AnyRankedTensor:$result
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
  AffineScope, AutomaticAllocationScope
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
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results), [{
      auto wg   = $_builder.getDenseI64ArrayAttr({2, 32, 16});
      auto bufs = $_builder.getDenseI64ArrayAttr({0, 65536, 0});

      $_state.addTypes(results);
      $_state.addAttribute("workgroupShape", wg);
      $_state.addAttribute("bufferSizesInBytes", bufs);
      // Do NOT set tileSizes here (absent means "use heuristic").

      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock(); // satisfy SizedRegion<1>
    }]>
  ];

//  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict-with-keyword (`->` type($results)^)? $body";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//
def Cinm_YieldOp : Cinm_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "Yield the result of a cinm.compute / cinm.compute_memref.";
  let description = [{
    Terminator for both `cinm.compute` (tensor results) and
    `cinm.compute_memref` (memref results). The operand types must match
    the parent op's result types exactly.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// ComputeMemRefOp 
//===----------------------------------------------------------------------===//
def Cinm_ComputeMemRefOp : Cinm_Base_Op<"compute_memref", [
  AffineScope, AutomaticAllocationScope
]> {
  let summary = "Container for bufferized compute; returns memrefs";
  let description = [{
    Memref-returning twin of `cinm.compute`. Same attributes and region
    semantics; used after bufferization to keep the container in the IR.

    Example:
      %y_mem = cinm.compute_memref attributes { ... } -> memref<...> {
        // bufferized body (alloc/subview/fill/…)
        cinm.yield %y_mem : memref<...>
      }
  }];

  let arguments = (ins
    DefaultValuedAttr<DenseI64ArrayAttr, "{2, 32, 16}">:$workgroupShape,
    DefaultValuedAttr<DenseI64ArrayAttr, "{0, 65536, 0}">:$bufferSizesInBytes,
    OptionalAttr<DenseI64ArrayAttr>:$tileSizes 
  );

  let results = (outs Variadic<AnyMemRef>:$results);

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    cnm::WorkgroupType getCnmWorkgroupType() {
      return cnm::WorkgroupType::get(getContext(), getWorkgroupShape());
    }
    long getWorkgroupSize() {
      auto wgShape = getWorkgroupShape();
      return std::reduce(wgShape.begin(), wgShape.end(), 1, std::multiplies<>());
    }
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results), [{
      auto wg   = $_builder.getDenseI64ArrayAttr({2, 32, 16});
      auto bufs = $_builder.getDenseI64ArrayAttr({0, 65536, 0});

      $_state.addTypes(results);
      $_state.addAttribute("workgroupShape", wg);
      $_state.addAttribute("bufferSizesInBytes", bufs);
      // Do NOT set tileSizes here (absent means "use heuristic").

      ::mlir::Region *region = $_state.addRegion();
      region->emplaceBlock(); // satisfy SizedRegion<1>
    }]>
  ];

  let assemblyFormat =
    "attr-dict-with-keyword (`->` type($results)^)? $body";
}

//===----------------------------------------------------------------------===//
// GemmOp - bufferized 
//===----------------------------------------------------------------------===//
def Cinm_GemmMemRefOp : Cinm_Op<"gemm_memref", []> {
  let summary = "GEMM on memrefs (C = A * B)";
  let description = [{
    Performs matrix multiplication on memrefs:
      out[m, n] = sum_k left[m, k] * right[k, n]

    Contract:
      - All operands are memref<2x?...xT> with the same element type T.
      - Shapes must be (M x K), (K x N), (M x N) (checked for static dims).
      - Op has no results; it writes into `out`.
  }];

  let arguments = (ins
    AnyMemRef:$left,   
    AnyMemRef:$right, 
    AnyMemRef:$out    
  );
  let results = (outs);

  let assemblyFormat =
    "$left `,` $right `,` $out attr-dict `:` type($left) `,` type($right) `,` type($out)";

}


//===----------------------------------------------------------------------===//
// GemvOp - bufferized 
//===----------------------------------------------------------------------===//
def Cinm_GemvMemRefOp : Cinm_Op<"gemv_memref", []> {
  let summary = "GEMV on memrefs (y = A * x)";
  let description = [{
    out[m] = sum_k left[m, k] * right[k]
  }];
  let arguments = (ins
    AnyMemRef:$left,   
    AnyMemRef:$right,  
    AnyMemRef:$out    
  );
  let results = (outs);
  let assemblyFormat =
    "$left `,` $right `,` $out attr-dict "
    "`:` type($left) `,` type($right) `,` type($out)";
}

def Cinm_AddMemRefOp : Cinm_Op<"add_memref", []> {
  let summary = "Elementwise add on memrefs: out[i...] = lhs[i...] + rhs[i...]";
  let description = [{
    Contract:
      - All operands are memrefs of the same rank and element type.
      - Shapes must match (checked for static dims).
      - No results; writes into `out`.
  }];

  let arguments = (ins
    AnyMemRef:$lhs,
    AnyMemRef:$rhs,
    AnyMemRef:$out
  );
  let results = (outs);

  let assemblyFormat =
    "$lhs `,` $rhs `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($out)";
}

def Cinm_ActivateOp : Cinm_Op<"op.activate",
  [Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<CinmTilingInterface>]> {

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
    AnyRankedTensor:$input
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$kind `(` $input `)` attr-dict `:` type($input)";
}

def Cinm_ActivateMemRefOp : Cinm_Op<"activate_memref", []> {
  let summary = "Elementwise activation on memrefs (out[i...] = act(in[i...]))";
  let description = [{
    Writes the activated values of `input` into `out`. Element types must match;
    ranks and shapes are expected to match (enforced by a verifier or patterns).
  }];

  let arguments = (ins
    Cinm_ActivationKindAttr:$kind,
    AnyMemRef:$input,
    AnyMemRef:$out
  );
  let results = (outs);

  let assemblyFormat =
    "$kind $input `,` $out attr-dict `:` type($input) `,` type($out)";

  let hasVerifier = 1;
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
    AnyRankedTensor:$src,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<Cinm_RoundingMode, "::mlir::cinm::RoundingMode::Nearest">:$rounding,
    DefaultValuedAttr<BoolAttr, "false">:$narrowRange
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$src attr-dict `:` type($src) `->` type($result)";
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
    AnyRankedTensor:$src,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
    "$src attr-dict `:` type($src) `->` type($result)";
}

def Cinm_QuantizeMemRefOp : Cinm_Op<"quantize_memref", []> {
  let summary = "Quantize memref (float -> integer), writes into out";
  let description = [{
    Per-tensor affine quantization of a memref:
      out[i...] = clamp(round(src[i...] / scale) + zeroPoint, qmin, qmax)

    Contract:
      - element_type(src) is floating, element_type(out) is integer
      - ranks & shapes match
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis,
    DefaultValuedAttr<Cinm_RoundingMode, "::mlir::cinm::RoundingMode::Nearest">:$rounding,
    DefaultValuedAttr<BoolAttr, "false">:$narrowRange
  );

  let results = (outs);

  let assemblyFormat =
    "$src `,` $out attr-dict `:` type($src) `,` type($out)";
}

def Cinm_DequantizeMemRefOp : Cinm_Op<"dequantize_memref", []> {
  let summary = "Dequantize memref (integer -> float), writes into out";
  let description = [{
    Per-tensor affine dequantization of a memref:
      out[i...] = (src[i...] - zeroPoint) * scale

    Contract:
      - element_type(src) is integer, element_type(out) is floating
      - ranks & shapes match
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$out,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    OptionalAttr<I64Attr>:$axis
  );

  let results = (outs);

  let assemblyFormat =
    "$src `,` $out attr-dict `:` type($src) `,` type($out)";
}

#endif
