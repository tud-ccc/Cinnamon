// RUN: cinm-opt %s | cinm-opt | FileCheck %s
// RUN: cinm-opt %s --mlir-print-op-generic | cinm-opt | FileCheck %s


// CHECK-LABEL: simple
func.func @simple(%t0: tensor<6x6xi32>, %t1 : tensor<6xf32> ) {
    %x = cinm.op.add %t0, %t0: tensor<6x6xi32>
    %y = cinm.op.sub %t0, %t0: tensor<6x6xi32>
    %z = cinm.op.max %t0: tensor<6x6xi32>
    %q = cinm.op.min %t1: tensor<6xf32>
    %a = arith.addi %z, %z : i32
    %k = arith.constant 62: i64
    %t, %s = cinm.op.topK %k (%y): tensor<6x6xi32> -> tensor<?xi32>, tensor<?xindex>
    %sum = cinm.op.reduce add (%y): tensor<6x6xi32> -> i32
    %product = cinm.op.reduce mul (%y): tensor<6x6xi32> -> i32 

    %scan = cinm.op.scan mul (%y): tensor<6x6xi32> 
    %scan2 = cinm.op.scan add (%y): tensor<6x6xi32>

    %othert = tensor.empty() : tensor<6x12xi32> 

    %sim1, %sim1i = cinm.op.simSearch cos 4 (%scan, %scan2) : tensor<6x6xi32>
    %sim2, %sim2i = cinm.op.simSearch dot 4 (%scan, %scan2) : tensor<6x6xi32>
    
    

    return
}

