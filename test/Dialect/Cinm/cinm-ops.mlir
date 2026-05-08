// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t00: tensor<6x6xi32>, %t10 : tensor<6xf32> , %m00: memref<6xf32>) attributes{cinm.available_platforms = [#cinm.host_platform]} {

    %d = cinm.compute_block (%t0 = %t00: tensor<6x6xi32>, %t1 = %t10 : tensor<6xf32> , %m0 = %m00: memref<6xf32>) -> tensor<6x6xi32> attributes { workgroupShape= array<i64: 2,4,4,2> } {
        %x = cinm.op.elementwise add %t0, %t0: tensor<6x6xi32>
        %y = cinm.op.elementwise sub %t0, %t0: tensor<6x6xi32>
        %y2 = cinm.op.elementwise div %t0, %t0: tensor<6x6xi32>
        %y8 = cinm.op.elementwise mul %t0, %t0 into %t0: tensor<6x6xi32> into tensor<6x6xi32>
        cinm.op.elementwise mul %t1, %t1 into %m0: tensor<6xf32> into memref<6xf32>

        %000 = cinm.op.elementwise exp %t0: tensor<6x6xi32>
        cinm.op.elementwise exp %t0 into %m0: tensor<6x6xi32> into memref<6xf32>

        %zx = cinm.op.reduce mul (%y) : tensor<6x6xi32> -> tensor<6xi32>
        %z = cinm.op.reduce mul (%zx) : tensor<6xi32> -> i32
        %i = arith.addi %z, %z : i32
        %k = arith.constant 62: i64
        %t, %s = cinm.op.topK %k (%y): tensor<6x6xi32> -> tensor<?xi32>, tensor<?xindex>

        %z4 = cinm.op.reduce add (%y): tensor<6x6xi32> -> tensor<6xi32>
        %z0 = cinm.op.reduce mul (%y): tensor<6x6xi32> -> tensor<6xi32>
        %z1 = cinm.op.reduce maxsi (%t0): tensor<6x6xi32> -> tensor<6xi32>
        %q2 = cinm.op.reduce minimumf (%t1): tensor<6xf32> -> f32


        %sqrts = cinm.op.elementwise sqrt %x: tensor<6x6xi32>
        %exps = cinm.op.elementwise exp %y: tensor<6x6xi32>

        %scan = cinm.op.scan mul (%y): tensor<6x6xi32>
        %scan2 = cinm.op.scan add (%y): tensor<6x6xi32>

        %othert = tensor.empty() : tensor<6x12xi32>

        %sim1, %sim1i = cinm.op.simSearch cos 4 (%scan, %scan2) : tensor<6x6xi32>
        %sim2, %sim2i = cinm.op.simSearch dot 4 (%scan, %scan2) : tensor<6x6xi32>

        %d2 = cinm.op.gemm %t0, %t0 : tensor<6x6xi32>, tensor<6x6xi32> -> tensor<6x6xi32>
        %a00 = tensor.empty (): tensor<6x4xi32>
        %a01 = tensor.empty (): tensor<4x22xi32>
        %a02 = tensor.empty (): tensor<6x22xi32>
        %d4 = cinm.op.gemm %a00, %a01 : tensor<6x4xi32>, tensor<4x22xi32> -> tensor<6x22xi32>
        %d3 = cinm.op.gemm %a00, %a01 plus %a02 {cinm.tile_sizes = array<i64: 2, 11, 2>}: tensor<6x4xi32>, tensor<4x22xi32> plus tensor<6x22xi32> -> tensor<6x22xi32>
        cinm.yield %d2: tensor<6x6xi32>
    }

    // different forms for compute
    cinm.compute_block on platform #cinm.host_platform () {
        cinm.yield
    }
    cinm.compute_block() attributes { maxDpuBufferSize = 64 } {
        cinm.yield
    }

    %a0, %b0 = cinm.compute_block() -> i64, i64 attributes {} {
        %cst = arith.constant 32: i64
        cinm.yield %cst, %cst: i64, i64
    }

    return
}
