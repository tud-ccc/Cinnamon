// RUN: cinm-opt %s --convert-memristor-to-func | cinm-opt | FileCheck %s

// CHECK: func.func nested @memristor_write_to_crossbar_i32(i64, memref<?x?xi32>)
// CHECK: func.func nested @memristor_gemm_i32(i64, memref<?x?xi32>, memref<?x?xi32>)
// CHECK: func.func nested @memristor_gevm_i32(i64, memref<?xi32>, memref<?xi32>)
// CHECK: func.func nested @memristor_barrier(i64)

// CHECK-LABEL: simple
// CHECK-SAME: (%[[arg0:.*]]: memref<6x6xi32>, %[[arg1:.*]]: memref<6x6xi32>, %[[arg2:.*]]: memref<6xi32>) { 
func.func @simple(%t0: memref<6x6xi32>, %t1 : memref<6x6xi32>, %t2 : memref<6xi32>) {
    %tile = arith.constant 0 : i64

    %rm0 = memref.alloc() : memref<6x6xi32>
    %rm1 = memref.alloc() : memref<6xi32>
// CHECK: %[[c0:.*]] = arith.constant 0 : i64 
// CHECK: %[[alloc0:.*]] = memref.alloc() : memref<6x6xi32> 
// CHECK: %[[alloc1:.*]] = memref.alloc() : memref<6xi32> 

// CHECK: %[[cast0:.*]] = memref.cast %[[arg1]] : memref<6x6xi32> to memref<?x?xi32>
// CHECK: call @memristor_write_to_crossbar_i32(%[[c0]], %[[cast0]]) : (i64, memref<?x?xi32>) -> ()
    memristor.write_to_crossbar %tile, %t1 : i64, memref<6x6xi32>
// CHECK: %[[cast0:.*]] = memref.cast %[[arg0]] : memref<6x6xi32> to memref<?x?xi32>
// CHECK: %[[cast_a0:.*]] = memref.cast %[[alloc0]] : memref<6x6xi32> to memref<?x?xi32>
// CHECK: call @memristor_gemm_i32(%[[c0]], %[[cast0]], %[[cast_a0]]) : (i64, memref<?x?xi32>, memref<?x?xi32>) -> ()
    memristor.gemm %tile, %t0, %rm0 : i64, memref<6x6xi32>, memref<6x6xi32>

// CHECK: %[[cast_a0:.*]] = memref.cast %[[alloc0]] : memref<6x6xi32> to memref<?x?xi32>
    memristor.write_to_crossbar %tile, %rm0 : i64, memref<6x6xi32>
// CHECK: %[[cast2:.*]] = memref.cast %[[arg2]] : memref<6xi32> to memref<?xi32>
// CHECK: %[[cast_a1:.*]] = memref.cast %[[alloc1]] : memref<6xi32> to memref<?xi32>
// CHECK: call @memristor_gevm_i32(%[[c0]], %[[cast2]], %[[cast_a1]]) : (i64, memref<?xi32>, memref<?xi32>) -> ()
    memristor.gevm %tile, %t2, %rm1 : i64, memref<6xi32>, memref<6xi32>

// CHECK: call @memristor_barrier(%[[c0]]) : (i64) -> ()
    memristor.barrier %tile : i64

    return
}