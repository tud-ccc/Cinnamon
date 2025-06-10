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


def AcquireDeviceOp : Cim_Op<"acquire_device", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let summary = "Acquire a cim device, returns id.";
    let description = [{Resereves resources needed to execute the requested computations on the device.}];

    let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$availableCrossbarCount, DefaultValuedAttr<BoolAttr, "false">:$isFullyScheduled);
    let results = (outs DeviceIdType:$deviceId);

    let assemblyFormat = "attr-dict `->` qualified(type($deviceId))";
}

def ReleaseDeviceOp : Cim_Op<"release_device", []> {
    let summary = "Release the device.";
    let description = [{}];

    let arguments = (ins DeviceIdType:$deviceId);
    let results = (outs);

    let assemblyFormat = "$deviceId attr-dict `:` qualified(type($deviceId))";
}

def AcquireCrossbarOp : Cim_Op<"acquire_crossbar", [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
    let summary = "Acquire a cim crossbar, returns id.";
    let description = [{Resereves crossbar on the device.}];

    let arguments = (ins DeviceIdType:$deviceId);
    let results = (outs CrossbarIdType:$crossbarId);

    let assemblyFormat = "$deviceId attr-dict `:` qualified(type($deviceId)) `->` qualified(type($crossbarId))";
}

def ReleaseCrossbarOp : Cim_Op<"release_crossbar", []> {
    let summary = "Release the crossbar.";
    let description = [{}];

    let arguments = (ins CrossbarIdType:$crossbarId);
    let results = (outs);

    let assemblyFormat = "$crossbarId attr-dict `:` qualified(type($crossbarId))";
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
    CPred<[{llvm::isa<TensorType>($_self)}]>,
    CPred<[{llvm::isa<cim::FutureType>($_self)}]>
]>> {}

def GemmOp : Cim_Op<"op.gemm", []> {
    let summary = "Matrix matrix multiplication.";
    let description = [{}];

    let arguments = (ins CrossbarIdType:$crossbarId, OpInput:$lhs, OpInput:$rhs);
    let results = (outs FutureType:$result);

    let assemblyFormat = "$crossbarId `,` $lhs `,` $rhs attr-dict `:` qualified(type($crossbarId)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";

    let hasVerifier = 1;
}

def GemvOp : Cim_Op<"op.gemv", []> {
    let summary = "Matrix vector multiplication.";
    let description = [{}];

    let arguments = (ins CrossbarIdType:$crossbarId, OpInput:$lhs, OpInput:$rhs);
    let results = (outs FutureType:$result);

    let assemblyFormat = "$crossbarId `,` $lhs `,` $rhs attr-dict `:` qualified(type($crossbarId)) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";

    let hasVerifier = 1;
}

#endif
