//===- Ops.td - Cnm dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Cnm dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CNM_OPS
#define CNM_OPS


include "cinm-mlir/Dialect/Cnm/IR/CnmBase.td"
include "cinm-mlir/Dialect/Cnm/IR/CnmTypes.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def WorkgroupOp : Cnm_Op<"workgroup"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins);
    let results = (outs WorkgroupType:$result);

    let hasCustomAssemblyFormat = 1;
}

def AllocOp : Cnm_Op<"alloc"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg);
    let results = (outs BufferType:$result);

    let hasCustomAssemblyFormat = 1;
}

def SetZeroOp: Cnm_Op<"set_zero"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins BufferType:$buffer);
    let results = (outs BufferType:$result);

    let hasCustomAssemblyFormat = 1;
}

def ScatterOp: Cnm_Op<"scatter"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins Builtin_RankedTensor:$input, BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$scatterMap);
    let results = (outs ScatterTokenType:$token);

    let hasCustomAssemblyFormat = 1;
}

def GatherOp: Cnm_Op<"gather"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins BufferType:$buffer, WorkgroupType:$wg, AffineMapAttr:$scatterMap);
    let results = (outs Builtin_RankedTensor:$output, GatherTokenType:$token);

    let hasCustomAssemblyFormat = 1;
}

def LaunchOp: Cnm_Op<"launch"> {
    let summary = "";
    let description = [{}];

    let arguments = (ins WorkgroupType:$wg, Variadic<AnyType>:$params);
    let results = (outs LaunchTokenType:$token);
    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
}

def TerminatorOp: Cnm_Op<"terminator", [Terminator]> {
    let summary = "";
    let description = [{}];

    let arguments = (ins);
    let results = (outs);

    let hasCustomAssemblyFormat = 1;
}

#endif
