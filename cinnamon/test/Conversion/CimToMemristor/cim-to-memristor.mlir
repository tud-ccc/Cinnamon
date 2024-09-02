// RUN: cinm-opt %s --cim-schedule-asap --convert-cim-to-memristor | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x3xi32>, %t1: tensor<3x6xi32>, %t2 : tensor<6xi32> ) {
// CHECK: %c0_i32 = arith.constant 0 : i32
    %device = cim.acquire -> !cim.deviceId

// CHECK: %0 = bufferization.alloc_tensor() : tensor<6x6xi32>
// CHECK: %1 = bufferization.to_memref %arg1 : memref<3x6xi32>
// CHECK: %2 = bufferization.to_memref %arg0 : memref<6x3xi32>
// CHECK: %3 = bufferization.to_memref %0 : memref<6x6xi32>
// CHECK: memristor.write_to_crossbar %c0_i32, %2 : i32, memref<6x3xi32>
// CHECK: memristor.gemm %c0_i32, %1, %3 : i32, memref<3x6xi32>, memref<6x6xi32>
// CHECK: memristor.barrier %c0_i32 : i32
    %r0 = cim.op.gemm %device, %t0, %t1 : !cim.deviceId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
// CHECK: %4 = bufferization.alloc_tensor() : tensor<6xi32>
// CHECK: %5 = bufferization.to_memref %arg2 : memref<6xi32>
// CHECK: %6 = bufferization.to_memref %0 : memref<6x6xi32>
// CHECK: %7 = bufferization.to_memref %4 : memref<6xi32>
// CHECK: memristor.write_to_crossbar %c0_i32, %6 : i32, memref<6x6xi32>
// CHECK: memristor.gevm %c0_i32, %5, %7 : i32, memref<6xi32>, memref<6xi32>
    %r1 = cim.op.gemv %device, %r0, %t2 : !cim.deviceId, !cim.future<6x6xi32>, tensor<6xi32> -> !cim.future<6xi32>

// CHECK: memristor.barrier %c0_i32 : i32
    %result = cim.barrier %r1 : !cim.future<6xi32> -> tensor<6xi32>

    cim.release %device : !cim.deviceId

    return
}
