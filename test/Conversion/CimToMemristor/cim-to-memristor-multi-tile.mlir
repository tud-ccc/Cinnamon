// RUN: cinm-opt %s --cim-schedule-asap --convert-cim-to-memristor | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x3xi32>, %t1: tensor<3x6xi32>, %t2 : tensor<6xi32> ) -> tensor<6x6xi32> {
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %c1_i32 = arith.constant 1 : i32
    %device = cim.acquire_device {availableCrossbarCount = 2 : i32} -> !cim.deviceId
    %crossbar0 = cim.acquire_crossbar %device : !cim.deviceId -> !cim.crossbarId
    %crossbar1 = cim.acquire_crossbar %device : !cim.deviceId -> !cim.crossbarId

// CHECK: %0 = bufferization.alloc_tensor() : tensor<6x6xi32>
// CHECK: %1 = bufferization.to_memref %arg0 : tensor<6x3xi32> to memref<6x3xi32>
// CHECK: %2 = bufferization.to_memref %arg1 : tensor<3x6xi32> to memref<3x6xi32>
// CHECK: %3 = bufferization.to_memref %0 : tensor<6x6xi32> to memref<6x6xi32>
// CHECK: memristor.write_to_crossbar %c1_i32, %2 : i32, memref<3x6xi32>
// CHECK: memristor.gemm %c1_i32, %1, %3 : i32, memref<6x3xi32>, memref<6x6xi32>
    %f0 = cim.op.gemm %crossbar0, %t0, %t1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r0 = cim.barrier %f0 : !cim.future<6x6xi32> -> tensor<6x6xi32>

// CHECK: %4 = bufferization.alloc_tensor() : tensor<6x6xi32>
// CHECK: %5 = bufferization.to_memref %arg0 : tensor<6x3xi32> to memref<6x3xi32>
// CHECK: %6 = bufferization.to_memref %arg1 : tensor<3x6xi32> to memref<3x6xi32>
// CHECK: %7 = bufferization.to_memref %4 : tensor<6x6xi32> to memref<6x6xi32>
// CHECK: memristor.write_to_crossbar %c0_i32, %6 : i32, memref<3x6xi32>
// CHECK: memristor.gemm %c0_i32, %5, %7 : i32, memref<6x3xi32>, memref<6x6xi32>
    %f1 = cim.op.gemm %crossbar1, %t0, %t1 : !cim.crossbarId, tensor<6x3xi32>, tensor<3x6xi32> -> !cim.future<6x6xi32>
    %r1 = cim.barrier %f1 : !cim.future<6x6xi32> -> tensor<6x6xi32>

// CHECK: memristor.barrier %c1_i32 : i32
// CHECK: memristor.barrier %c0_i32 : i32

// CHECK: %8 = bufferization.alloc_tensor() : tensor<6x6xi32>
// CHECK: %9 = bufferization.to_memref %0 : tensor<6x6xi32> to memref<6x6xi32>
// CHECK: %10 = bufferization.to_memref %4 : tensor<6x6xi32> to memref<6x6xi32>
// CHECK: %11 = bufferization.to_memref %8 : tensor<6x6xi32> to memref<6x6xi32>
// CHECK: memristor.write_to_crossbar %c1_i32, %10 : i32, memref<6x6xi32>
// CHECK: memristor.gemm %c1_i32, %9, %11 : i32, memref<6x6xi32>, memref<6x6xi32>
    %f2 = cim.op.gemm %crossbar0, %r0, %r1 : !cim.crossbarId, tensor<6x6xi32>, tensor<6x6xi32> -> !cim.future<6x6xi32>

// CHECK: memristor.barrier %c1_i32 : i32
    %result = cim.barrier %f2 : !cim.future<6x6xi32> -> tensor<6x6xi32>

    cim.release_crossbar %crossbar0 : !cim.crossbarId
    cim.release_crossbar %crossbar1 : !cim.crossbarId
    cim.release_device %device : !cim.deviceId

    return %result : tensor<6x6xi32>
}