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
include "cinm-mlir/Dialect/Cinm/Interfaces/TilingInterface.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"




// Op templates

defvar AnyScalar = AnyType; // todo

// no implicit broadcast
class Cinm_TTT_Op<string name, list<Trait> traits = []>
    : Cinm_Op<name, traits # [SameOperandsAndResultType]> {

    let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Cinm_Bitwise_Op<string name, list<Trait> traits = []>
    : Cinm_Op<name, traits> {
    let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
    let results = (outs AnyRankedTensor:$result);
}








// Concrete op definitions

class TypeMatchesElementType<string tensorTy, string toInfer>:
TypesMatchWith<"output type matches element type of input", toInfer, tensorTy,
              "::llvm::cast<RankedTensorType>($_self).getElementType()">;

def Cinm_AddOp : Cinm_TTT_Op<"op.add", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor addition operation";
  let description = [{
    TODO
  }];
}
def Cinm_SubOp : Cinm_TTT_Op<"op.sub", [Pure,  DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor subtraction operation";
  let description = [{
    TODO
  }];
}
// these two aren't in the paper
def Cinm_MulOp : Cinm_TTT_Op<"op.mul", [Commutative, Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Hadamard product (element-wise multiplication)";
  let description = [{
    TODO
  }];
}

def Cinm_DivOp : Cinm_TTT_Op<"op.div", [Pure, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Tensor division operation";
  let description = [{
    TODO
  }];
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

def Cinm_GemvOp : Cinm_Op<"op.gemv", [Pure, InferShapedTypeOpAdaptor, DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Generalized Matrix Vector Multiplication";
  let description = [{
    TODO
  }];
  let arguments = (ins 2DTensorOf<[AnyType]>:$left, 1DTensorOf<[AnyType]>:$right);
  let results = (outs 1DTensorOf<[AnyType]>:$result);
  let assemblyFormat = "$left `,` $right attr-dict `:` `(` type($left) `,` type($right) `)` `->` type($result)";
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
    [Pure, TypeMatchesElementType<"result", "input">,
    DeclareOpInterfaceMethods<CinmTilingInterface>]> {
  let summary = "Reduce the tensor according to a given binary operator.";

  let description = [{

  }];

  let arguments = (ins
    Cinm_ReduceMethodAttr:$method,
    AnyRankedTensor:$input
  );

  let results = (
    outs
    AnyScalar:$result
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$method `(` $input `)` attr-dict `:` type($input)";
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
//  IsolatedFromAbove //, Pure
]> {
  let summary = "TODO";
  let description = [{

    %rbuf = cinm.compute attributes { ... } -> tensor<...> {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %arg0, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
  }];

  /// These attributes are used for lowering of the contained cinm ops into CNM.
  let arguments = (ins DefaultValuedAttr<DenseI64ArrayAttr, "{2, 32, 16}">:$workgroupShape,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{0, 65536, 0}">:$bufferSizesInBytes);
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
//  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "attr-dict-with-keyword (`->` type($results)^)? $body";
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def Cinm_YieldOp : Cinm_Op<"yield", [Pure, HasParent<"ComputeOp">,
                                     ReturnLike, Terminator]> {
  let summary = "Yield the result of a cinm.compute operator.";
  let description = [{
      TODO
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  //let hasVerifier = 1;
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

}


#endif
