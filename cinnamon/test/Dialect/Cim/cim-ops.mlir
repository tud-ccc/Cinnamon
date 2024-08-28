// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x3xi32>, %t1: tensor<3x6xi32>, %t2 : tensor<6xi32> ) {
    %device = cim.acquire -> !cim.deviceId

    %r0 = cim.op.gemm %device, %t0, %t1 : !cim.deviceId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r1 = cim.op.gemv %device, %r0, %t2 : !cim.deviceId, !cim.future<6x6xi32>, tensor<6xi32> -> !cim.future<6xi32>

    %result = cim.barrier %r1 : !cim.future<6xi32> -> tensor<6xi32>

    cim.release %device : !cim.deviceId

    return
}
