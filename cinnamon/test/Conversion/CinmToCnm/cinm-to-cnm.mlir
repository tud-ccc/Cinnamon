// RUN: cinm-opt --convert-cinm-to-cnm %s | cinm-opt | FileCheck %s

// CHECK-LABEL: mm_dimm8_nopt
    func.func @mm_dimm8_nopt(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

// CHECK: %[[wg:.*]] = cnm.workgroup : !cnm.workgroup<8x128x1>
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<128x1024xi32>
// CHECK: %[[transposed:.*]] = linalg.transpose ins(%arg1 : tensor<1024x128xi32>) outs(%[[empty]] : tensor<128x1024xi32>) permutation = [1, 0] 
// CHECK: %[[ba:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on 8x128x1, level 0>
// CHECK: %[[bb:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on 8x128x1, level 0>
// CHECK: %[[bc:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<i32 on 8x128x1, level 0>
// CHECK: cnm.scatter %arg0 into %[[ba]][#map] of %[[wg]] : tensor<8x1024xi32> into !cnm.buffer<1024xi32 on 8x128x1, level 0>
// CHECK: cnm.scatter %[[transposed]] into %[[bb]][#map1] of %[[wg]] : tensor<128x1024xi32> into !cnm.buffer<1024xi32 on 8x128x1, level 0>
// CHECK: %[[cst0:.*]] = arith.constant dense<0> : tensor<8x128xi32>
// CHECK: cnm.scatter %[[cst0]] into %[[bc]][#map2] of %[[wg]] : tensor<8x128xi32> into !cnm.buffer<i32 on 8x128x1, level 0>
// CHECK: cnm.launch %[[wg]] in(%[[ba]], %[[bb]] : !cnm.buffer<1024xi32 on 8x128x1, level 0>, !cnm.buffer<1024xi32 on 8x128x1, level 0>) out(%[[bc]] : !cnm.buffer<i32 on 8x128x1, level 0>) on !cnm.workgroup<8x128x1> {
// CHECK: ^bb0(%[[arg2:.*]]: memref<1024xi32>, %[[arg3:.*]]: memref<1024xi32>, %[[arg4:.*]]: memref<i32>):
// CHECK:    linalg.contract
// CHECK: }
// CHECK: %[[emptyres:.*]] = tensor.empty() : tensor<8x128xi32>
// CHECK: %{{.*}} = cnm.gather %[[bc]][#map2] of %[[wg]] into %[[emptyres]] : !cnm.buffer<i32 on 8x128x1, level 0> into tensor<8x128xi32>
// CHECK: cnm.free_workgroup %[[wg]] : !cnm.workgroup<8x128x1>
        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<8x128xi32> {
            %r = cinm.op.gemm %A, %B: (tensor<8x1024xi32>, tensor<1024x128xi32>) -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }

