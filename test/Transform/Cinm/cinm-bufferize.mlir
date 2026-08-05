// RUN: cinm-opt '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' %s --split-input-file | cinm-opt | FileCheck %s

// CHECK-LABEL: gemm
    func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

    // CHECK:  cinm.compute_block ({{.*}}) -> memref<8x128xi32>
    // CHECK:   %[[res:.*]] = memref.alloc() : memref<8x128xi32>
    // CHECK:   %[[c0:.*]] = arith.constant 0 : i32
    // CHECK:   linalg.fill ins(%[[c0]] : i32) outs(%[[res]] : memref<8x128xi32>)
    // CHECK:   cinm.op.gemm %{{.*}}, %{{.*}} into %[[res]] : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    // CHECK:   cinm.yield %[[res]] : memref<8x128xi32>
    // CHECK: }
        // %r0 = cinm.compute  -> tensor<8x128xi32>{
        //     %r = cinm.op.gemm %A, %B: tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
        //     cinm.yield %r : tensor<8x128xi32>
        // }
        %r0 = cinm.compute_block (%a = %A : tensor<8x1024xi32>, %b = %B : tensor<1024x128xi32>) -> tensor<8x128xi32> {
            %r = cinm.op.gemm %a, %b: tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }


// CHECK-LABEL: gemv
    func.func @gemv(%A: tensor<8x1024xi32>, %B: tensor<1024xi32>) -> tensor<8xi32> {

    // CHECK:  cinm.compute_block ({{.*}}) -> memref<8xi32>
    // CHECK:   %[[res:.*]] = memref.alloc() : memref<8xi32>
    // CHECK:   %[[c0:.*]] = arith.constant 0 : i32
    // CHECK:   linalg.fill ins(%[[c0]] : i32) outs(%[[res]] : memref<8xi32>)
    // CHECK:   cinm.op.gemv %{{.*}}, %{{.*}} into %[[res]] : memref<8x1024xi32>, memref<1024xi32> into memref<8xi32>
    // CHECK:   cinm.yield %[[res]] : memref<8xi32>
    // CHECK: }
        %r0 = cinm.compute_block (%a = %A : tensor<8x1024xi32>, %b = %B : tensor<1024xi32>) -> tensor<8xi32> attributes { workgroupShape=array<i64: 1, 8, 1> } {
            %r = cinm.op.gemv %a, %b : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
            cinm.yield %r : tensor<8xi32>
        }
        func.return %r0 : tensor<8xi32>
    }


// CHECK-LABEL: eltwise
func.func @eltwise(%t0: tensor<6x6xi32>, %t1 : tensor<6xf32> , %m0: memref<6xf32>) {
    // CHECK: %[[alloc:.*]] = memref.alloc
    // CHECK: cinm.op.elementwise add %[[a0:.*]], %[[a0]] into %[[alloc]] : memref<6x6xi32> into memref<6x6xi32>
    %x = cinm.op.elementwise add %t0, %t0: tensor<6x6xi32>
    return
}


// CHECK-LABEL: eltwise_into
func.func @eltwise_into(%t0: tensor<6x6xi32>, %t1 : tensor<6xf32> , %m0: memref<6xf32>) {
    // CHECK-NOT: %[[alloc:.*]] = memref.alloc
    // CHECK: cinm.op.elementwise mul %[[a0:.*]], %[[a0]] into %{{.*}} : memref<6xf32> into memref<6xf32>
    cinm.op.elementwise mul %t1, %t1 into %m0: tensor<6xf32> into memref<6xf32>
    return
}
