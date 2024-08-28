// RUN: cinm-opt %s --cim-schedule-alap | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x3xi32>, %t1: tensor<3x6xi32>, %t2: tensor<6x6xi32>, %t3 : tensor<6x6xi32> ) {
// CHECK: %0 = cim.acquire -> !cim.deviceId
    %device = cim.acquire -> !cim.deviceId

// CHECK: %1 = cim.op.gemm %0, %arg0, %arg1 : !cim.deviceId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r0 = cim.op.gemm %device, %t0, %t1 : !cim.deviceId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
// CHECK: %2 = cim.op.gemm %0, %arg2, %arg3 : !cim.deviceId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
    %r1 = cim.op.gemm %device, %t2, %t3 : !cim.deviceId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>

// CHECK: %3 = cim.barrier %1 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %4 = cim.barrier %2 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %5 = cim.op.gemm %0, %3, %4 : !cim.deviceId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
    %r2 = cim.op.gemm %device, %r0, %r1 : !cim.deviceId, !cim.future<6x6xi32>, !cim.future<6x6xi32> -> !cim.future<6x6xi32>
// CHECK: %6 = cim.op.gemm %0, %3, %4 : !cim.deviceId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
    %r3 = cim.op.gemm %device, %r0, %r1 : !cim.deviceId, !cim.future<6x6xi32>, !cim.future<6x6xi32> -> !cim.future<6x6xi32>

// CHECK: %7 = cim.barrier %6 : !cim.future<6x6xi32> -> tensor<6x6xi32>
    %result = cim.barrier %r3 : !cim.future<6x6xi32> -> tensor<6x6xi32>

// CHECK: cim.release %0 : !cim.deviceId
    cim.release %device : !cim.deviceId

    return
}
