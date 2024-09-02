// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: memref<6x6xi32>, %t1 : memref<6x6xi32>, %t2 : memref<6xi32>) {
    %tile = arith.constant 0 : i64

    %rt0 = bufferization.alloc_tensor() : tensor<6x6xi32>
    %rt1 = bufferization.alloc_tensor() : tensor<6xi32>

    %rm0 = bufferization.to_memref %rt0 : memref<6x6xi32>
    %rm1 = bufferization.to_memref %rt1 : memref<6xi32>

    memristor.write_to_crossbar %tile, %t1 : i64, memref<6x6xi32>
    memristor.gemm %tile, %t0, %rm0 : i64, memref<6x6xi32>, memref<6x6xi32>

    memristor.write_to_crossbar %tile, %rm0 : i64, memref<6x6xi32>
    memristor.gevm %tile, %t2, %rm1 : i64, memref<6xi32>, memref<6xi32>

    memristor.barrier %tile : i64

    return
}

