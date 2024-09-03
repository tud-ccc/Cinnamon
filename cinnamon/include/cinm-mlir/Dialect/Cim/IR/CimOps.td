//===- Ops.td - Cim dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Cim dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_OPS
#define CIM_OPS


include "cinm-mlir/Dialect/Cim/IR/CimBase.td"
include "cinm-mlir/Dialect/Cim/IR/CimTypes.td"

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


def AcquireOp : Cim_Op<"acquire", []> {
    let summary = "Acquire a (portion of a) cim device, returns id.";
    let description = [{Resereves resources needed to execute the requested computations on the device.
                       Depending on device capabilities, this may reserve the entire device or only a subset.}];

    let arguments = (ins);
    let results = (outs DeviceIdType:$deviceId);

    let assemblyFormat = "attr-dict `->` qualified(type($deviceId))";
}

def ReleaseOp : Cim_Op<"release", []> {
    let summary = "Release the device.";
    let description = [{}];

    let arguments = (ins DeviceIdType:$deviceId);
    let results = (outs);

    let assemblyFormat = "$deviceId attr-dict `:` qualified(type($deviceId))";
}

def BarrierOp : Cim_Op<"barrier", []> {
    let summary = "Wait for dependent computations to finish.";
    let description = [{Blocks until the value of the associated future is present.
                       This operation can also be used to enforce a particular scheduling on the execution graph.}];

    let arguments = (ins FutureType:$value);
    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` type($result)";
}

def OpInput : TypeConstraint<Or<[
    CPred<[{$_self.isa<TensorType>()}]>,
    CPred<[{$_self.isa<cim::FutureType>()}]>
]>> {}

def GemmOp : Cim_Op<"op.gemm", []> {
    let summary = "Matrix matrix multiplication.";
    let description = [{}];

    let arguments = (ins DeviceIdType:$deviceId, OpInput:$lhs, OpInput:$rhs);
    let results = (outs FutureType:$result);

    let assemblyFormat = "$deviceId `,` $lhs `,` $rhs attr-dict `:` qualified(type($deviceId)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";

    let hasVerifier = 1;
}

def GemvOp : Cim_Op<"op.gemv", []> {
    let summary = "Matrix vector multiplication.";
    let description = [{}];

    let arguments = (ins DeviceIdType:$deviceId, OpInput:$lhs, OpInput:$rhs);
    let results = (outs FutureType:$result);

    let assemblyFormat = "$deviceId `,` $lhs `,` $rhs attr-dict `:` qualified(type($deviceId)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";

    let hasVerifier = 1;
}

#endif
