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

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"




// Op templates

defvar AnyScalar = AnyType; // todo 

// no implicit broadcast
class Cinm_TTT_Op<string name, list<Trait> traits = []> 
    : Cinm_Op<name, traits # [SameOperandsAndResultType]> {

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Cinm_Bitwise_Op<string name, list<Trait> traits = []> 
    : Cinm_Op<name, traits> {
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}








// Concrete op definitions


def MinOp: Cinm_Op<"op.min", [Commutative, Pure]> {
    let arguments = (ins AnyTensor:$input);  
    let results = (outs AnyScalar:$result);
    let summary = "Return the min value of the input tensor";
    let hasCustomAssemblyFormat = 1;
}

def MaxOp: Cinm_Op<"op.max", [Commutative, Pure]> {
    let arguments = (ins AnyTensor:$input);  
    let results = (outs AnyScalar:$result);

    let summary = "Return the max value of the input tensor";
    let hasCustomAssemblyFormat = 1;
}

def Cinm_AddOp : Cinm_TTT_Op<"op.add", [Commutative, Pure]> {
  let summary = "Tensor addition operation";
  let description = [{
    TODO
  }];
}
def Cinm_SubOp : Cinm_TTT_Op<"op.sub", [Pure]> {
  let summary = "Tensor subtraction operation";
  let description = [{
    TODO
  }];
}
// these two aren't in the paper
def Cinm_MulOp : Cinm_TTT_Op<"op.mul", [Commutative, Pure]> {
  let summary = "Hadamard product (element-wise multiplication)";
  let description = [{
    TODO
  }];
}

def Cinm_DivOp : Cinm_TTT_Op<"div", []> {
  let summary = "AnyTensor division operation";
  let description = [{
    TODO
  }];
}

def Cinm_GemmOp : Cinm_Op<"op.gemm", [Pure]> {
  let summary = "Generalized Matrix Matrix Multiplication operation";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyTensor:$left, AnyTensor:$right);
  let results = (outs AnyTensor:$result);
}

def Cinm_GemvOp : Cinm_Op<"op.gemv", [Pure]> {
  let summary = "Generalized Matrix Vector Multiplication";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyTensor:$left, AnyTensor:$right);
  let results = (outs AnyTensor:$result);
}


/// we infer the shape because verifying correctness of a
/// hardcoded return type is as hard as inferring it.
def Cinm_TransposeOp : Cinm_Op<"op.transpose", [InferShapedTypeOpAdaptor, Pure]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    AnyTensor:$input1,
    AnyTensor:$perms // TODO 
  );

  let results = (
    outs AnyTensor:$output
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
    AnyTensor:$input
  );

  let results = (
    outs 
    AnyTensor:$result
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$method `(` $input `)` attr-dict `:` type($input)";
}

def Cinm_ReduceOp : Cinm_Op<"op.reduce", [Pure]> {
  let summary = "Reduce the tensor.";

  let description = [{

  }];

  let arguments = (ins
    Cinm_ReduceMethodAttr:$method,
    AnyTensor:$input
  );

  let results = (
    outs 
    AnyScalar:$result
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
    AnyTensor:$input,
    I64:$k // size of result tensor
  );

  let results = (
    outs 
    // these have dynamic sizes and rank 1
    AnyTensor:$resultValues,
    TensorOf<[Builtin_Index]>:$resultIndices
  );

  let extraClassDeclaration = [{
  }];

  let assemblyFormat = "$k `(` $input `)` attr-dict `:` type($input) `->` type($resultValues) `,` type($resultIndices)";
}

def Cinm_SimSearchOp : Cinm_Op<"op.simSearch", [InferShapedTypeOpAdaptor, Pure]> {
  let summary = "Similarity search between two tensors";

  let description = [{

  }];

  let arguments = (ins
    // two tensor parameters to search, TODO must have same shape
    Cinm_SimilarityMetricAttr:$metric,
    AnyTensor:$left,
    AnyTensor:$right,
    Builtin_IntegerAttr:$k // size of result tensors (maybe max size)
  );

  let results = (
    outs 
    // these have dynamic sizes and rank 1 
    AnyTensor:$resultValues,
    AnyTensor:$resultIndices
  );

  let extraClassDeclaration = [{
  }];

  let hasCustomAssemblyFormat = 1;
}

def Cinm_MergePartialOp : Cinm_Op<"op.mergePartial", [Pure]> {
  let summary = "Merge partial results of a sim search (?)";

  let description = [{

  }];

  let arguments = (ins
    // todo there are 2 enum parameters, what are they?
    // What are constraints on these tensors?
    // Should we use an out tensor? (ie is shape known?)
    AnyTensor:$left,
    AnyTensor:$right
  );

  let results = (
    outs
    AnyTensor:$result
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



def Cinm_ComputeOp : Cinm_Op<"compute", [
  AffineScope, AutomaticAllocationScope, 
  IsolatedFromAbove //, Pure
]> {
  let summary = "TODO";
  let description = [{

    %rbuf = cinm.compute(%arg0 = %inpt : tensor<...>) -> tensor<...> {
      %flt = arith.constant <"...">: tensor<...>
      %conv = cinm.op.gemm %arg0, %flt: tensor<...>, tensor<...>
      cinm.yield %conv : tensor<...>
    }
  }];

  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$result); // todo should we support returning multiple?
  let regions = (region AnyRegion:$body);


  let extraClassDeclaration = [{

  }];
  let hasCustomAssemblyFormat = 1;
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
  let assemblyFormat = "$operands attr-dict `:` type($operands)";

}


#endif
