// RUN: cinm-opt %s --cim-schedule-alap | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x3xi32>, %t1: tensor<3x6xi32>, %t2 : tensor<6xi32> ) -> tensor<6x6xi32> {

// CHECK: %cim_dev = cim.acquire_device {availableCrossbarCount = 4 : i32, isFullyScheduled = true} -> !cim.deviceId
// CHECK: %cim_cbr = cim.acquire_crossbar %cim_dev : !cim.deviceId -> !cim.crossbarId
// CHECK: %cim_cbr_0 = cim.acquire_crossbar %cim_dev : !cim.deviceId -> !cim.crossbarId
// CHECK: %cim_cbr_1 = cim.acquire_crossbar %cim_dev : !cim.deviceId -> !cim.crossbarId
// CHECK: %cim_cbr_2 = cim.acquire_crossbar %cim_dev : !cim.deviceId -> !cim.crossbarId
    %device = cim.acquire_device {availableCrossbarCount = 4 : i32} -> !cim.deviceId
    %crossbar0 = cim.acquire_crossbar %device : !cim.deviceId -> !cim.crossbarId

    // The following operations have this dependency tree:
    // r0  r1
    //  \  /
    //   r2  r3
    //    \ /
    //   result

// CHECK: %0 = cim.op.gemm %cim_cbr_2, %arg0, %arg1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
// CHECK: %1 = cim.op.gemm %cim_cbr_1, %arg0, %arg1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
// CHECK: %2 = cim.barrier %0 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %3 = cim.barrier %1 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %4 = cim.op.gemm %cim_cbr_2, %2, %3 : !cim.crossbarId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
// CHECK: %5 = cim.op.gemm %cim_cbr_1, %arg0, %arg1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
// CHECK: %6 = cim.barrier %4 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %7 = cim.barrier %5 : !cim.future<6x6xi32> -> tensor<6x6xi32>
// CHECK: %8 = cim.op.gemm %cim_cbr_2, %6, %7 : !cim.crossbarId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
// CHECK: %9 = cim.barrier %8 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    %f0 = cim.op.gemm %crossbar0, %t0, %t1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r0 = cim.barrier %f0 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    %f1 = cim.op.gemm %crossbar0, %t0, %t1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r1 = cim.barrier %f1 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    %f2 = cim.op.gemm %crossbar0, %r0, %r1 : !cim.crossbarId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
    %r2 = cim.barrier %f2 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    %f3 = cim.op.gemm %crossbar0, %t0, %t1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r3 = cim.barrier %f3 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    %fr = cim.op.gemm %crossbar0, %r2, %r3 : !cim.crossbarId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>
    %result = cim.barrier %fr : !cim.future<6x6xi32> -> tensor<6x6xi32>

// CHECK: cim.release_crossbar %cim_cbr_2 : !cim.crossbarId
// CHECK: cim.release_crossbar %cim_cbr_1 : !cim.crossbarId
// CHECK: cim.release_crossbar %cim_cbr_0 : !cim.crossbarId
// CHECK: cim.release_crossbar %cim_cbr : !cim.crossbarId
// CHECK: cim.release_device %cim_dev : !cim.deviceId

    cim.release_crossbar %crossbar0 : !cim.crossbarId
    cim.release_device %device : !cim.deviceId

// CHECK: return %9 : tensor<6x6xi32>

    return %result : tensor<6x6xi32>
}
