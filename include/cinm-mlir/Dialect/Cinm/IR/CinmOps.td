//===- Ops.td - Cinm dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Cinm dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_OPS
#define CINM_OPS


include "cinm-mlir/Dialect/Cinm/IR/CinmBase.td"
include "cinm-mlir/Dialect/Cinm/IR/CinmTypes.td"

include "mlir/IR/EnumAttr.td"
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


def MinOp: Cinm_Op<"min", [Commutative, Pure]> {
    let arguments = (ins AnyTensor:$input);  
    let results = (outs AnyScalar:$result);
    let summary = "Return the min value of the input tensor";
    let hasCustomAssemblyFormat = 1;
}

def MaxOp: Cinm_Op<"max", [Commutative, Pure]> {
    let arguments = (ins AnyTensor:$input);  
    let results = (outs AnyScalar:$result);

    let summary = "Return the max value of the input tensor";
    let hasCustomAssemblyFormat = 1;
}

def Cinm_AddOp : Cinm_TTT_Op<"add", [Commutative, Pure]> {
  let summary = "Tensor addition operation";
  let description = [{
    TODO
  }];
}
def Cinm_SubOp : Cinm_TTT_Op<"sub", [Pure]> {
  let summary = "Tensor subtraction operation";
  let description = [{
    TODO
  }];
}
// these two aren't in the paper
def Cinm_MulOp : Cinm_TTT_Op<"mul", [Commutative, Pure]> {
  let summary = "Hadamard product (element-wise multiplication)";
  let description = [{
    TODO
  }];
}

// def Cinm_DivOp : Cinm_TTT_Op<"div", []> {
//   let summary = "AnyTensor division operation";
//   let description = [{
//     TODO
//   }];
// }

def Cinm_GemmOp : Cinm_Op<"gemm", [Pure]> {
  let summary = "Generalized Matrix Matrix Multiplication operation";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyTensor:$left, AnyTensor:$right);
  let results = (outs AnyTensor:$result);
}

def Cinm_GemvOp : Cinm_Op<"gemv", [Pure]> {
  let summary = "Generalized Matrix Vector Multiplication";
  let description = [{
    TODO
  }];
  let arguments = (ins AnyTensor:$left, AnyTensor:$right);
  let results = (outs AnyTensor:$result);
}


/// we infer the shape because verifying correctness of a
/// hardcoded return type is as hard as inferring it.
def Cinm_TransposeOp : Cinm_Op<"transpose", [InferShapedTypeOpAdaptor, Pure]> {
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


#endif
