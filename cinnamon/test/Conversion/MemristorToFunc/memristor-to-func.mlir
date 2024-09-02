// RUN: cinm-opt %s --convert-memristor-to-func | cinm-opt | FileCheck %s

// CHECK: func.func nested @memristor_write_to_crossbar_i32(i64, memref<6x6xi32>)
// CHECK: func.func nested @memristor_gemm_i32(i64, memref<6x6xi32>, memref<6x6xi32>)
// CHECK: func.func nested @memristor_gevm_i32(i64, memref<6xi32>, memref<6xi32>)
// CHECK: func.func nested @memristor_barrier(i64)

// CHECK-LABEL: simple
func.func @simple(%t0: memref<6x6xi32>, %t1 : memref<6x6xi32>, %t2 : memref<6xi32>) {
    %tile = arith.constant 0 : i64

    %rt0 = bufferization.alloc_tensor() : tensor<6x6xi32>
    %rt1 = bufferization.alloc_tensor() : tensor<6xi32>

    %rm0 = bufferization.to_memref %rt0 : memref<6x6xi32>
    %rm1 = bufferization.to_memref %rt1 : memref<6xi32>

// CHECK: call @memristor_write_to_crossbar_i32(%c0_i64, %arg1) : (i64, memref<6x6xi32>) -> ()
    memristor.write_to_crossbar %tile, %t1 : i64, memref<6x6xi32>
// CHECK: call @memristor_gemm_i32(%c0_i64, %arg0, %2) : (i64, memref<6x6xi32>, memref<6x6xi32>) -> ()
    memristor.gemm %tile, %t0, %rm0 : i64, memref<6x6xi32>, memref<6x6xi32>

// CHECK: call @memristor_write_to_crossbar_i32(%c0_i64, %2) : (i64, memref<6x6xi32>) -> ()
    memristor.write_to_crossbar %tile, %rm0 : i64, memref<6x6xi32>
// CHECK: call @memristor_gevm_i32(%c0_i64, %arg2, %3) : (i64, memref<6xi32>, memref<6xi32>) -> ()
    memristor.gevm %tile, %t2, %rm1 : i64, memref<6xi32>, memref<6xi32>

// CHECK: call @memristor_barrier(%c0_i64) : (i64) -> ()
    memristor.barrier %tile : i64

    return
}

