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


// A block that inserts its result into one of its operands bufferizes in place,
// even when the destination is carried by an enclosing loop. This is the shape
// --cinm-expand-compute-scope produces.

// CHECK-LABEL: insert_slice_in_block
func.func @insert_slice_in_block(%A: tensor<8x1024xi32>, %B: tensor<1024xi32>) -> tensor<64xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %init = tensor.empty() : tensor<64xi32>

    // CHECK:     %[[OUT:.*]] = memref.alloc() {{.*}}: memref<64xi32>
    // CHECK:     scf.for %{{.*}} = {{.*}} iter_args(%[[ACC:.*]] = %[[OUT]])
    // CHECK:       cinm.compute_block ({{.*}}%[[D:.*]] = %[[ACC]] : memref<64xi32>) -> memref<64xi32>
    // CHECK:         %[[SV:.*]] = memref.subview %[[D]][0] [8] [1]
    // CHECK:         memref.copy %{{.*}}, %[[SV]]
    // CHECK:         cinm.yield %[[D]] : memref<64xi32>
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> tensor<64xi32> {
      %t = cinm.compute_block (%a = %A : tensor<8x1024xi32>, %b = %B : tensor<1024xi32>, %d = %acc : tensor<64xi32>) -> tensor<64xi32> {
        %v = cinm.op.gemv %a, %b : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
        %ins = tensor.insert_slice %v into %d[0] [8] [1] : tensor<8xi32> into tensor<64xi32>
        cinm.yield %ins : tensor<64xi32>
      }
      scf.yield %t : tensor<64xi32>
    }
    func.return %r : tensor<64xi32>
}
