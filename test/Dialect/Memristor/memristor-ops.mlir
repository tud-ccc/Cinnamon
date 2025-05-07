// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: memref<6x6xi32>, %t1 : memref<6x6xi32>, %t2 : memref<6xi32>) {
    %tile = arith.constant 0 : i64

    %rm0 = memref.alloc() : memref<6x6xi32>
    %rm1 = memref.alloc() : memref<6xi32>

    memristor.write_to_crossbar %tile, %t1 : i64, memref<6x6xi32>
    memristor.gemm %tile, %t0, %rm0 : i64, memref<6x6xi32>, memref<6x6xi32>

    memristor.write_to_crossbar %tile, %rm0 : i64, memref<6x6xi32>
    memristor.gevm %tile, %t2, %rm1 : i64, memref<6xi32>, memref<6xi32>

    memristor.barrier %tile : i64

    return
}

