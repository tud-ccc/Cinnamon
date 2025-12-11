// RUN: cinm-opt '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' %s | cinm-opt | FileCheck %s

// CHECK-LABEL: gemm
    func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

    // CHECK:  cinm.compute attributes {workgroupShape = array<i64: 8, 128, 1>} -> memref<8x128xi32> {
    // CHECK:   %[[res:.*]] = memref.alloc() : memref<8x128xi32>
    // CHECK:   %[[c0:.*]] = arith.constant 0 : i32
    // CHECK:   linalg.fill ins(%[[c0]] : i32) outs(%[[res]] : memref<8x128xi32>)
    // CHECK:   cinm.op.gemm %{{.*}}, %{{.*}} into %[[res]] : memref<8x1024xi32>, memref<1024x128xi32> into memref<8x128xi32>
    // CHECK:   cinm.yield %[[res]] : memref<8x128xi32>
    // CHECK: }
        %r0 = cinm.compute attributes { workgroupShape=array<i64: 8, 128, 1> } -> tensor<8x128xi32> {
            %r = cinm.op.gemm %A, %B: tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }


// CHECK-LABEL: gemv
    func.func @gemv(%A: tensor<8x1024xi32>, %B: tensor<1024xi32>) -> tensor<8xi32> {

    // CHECK:  cinm.compute attributes {workgroupShape = array<i64: 1, 8, 1>} -> memref<8xi32> {
    // CHECK:   %[[res:.*]] = memref.alloc() : memref<8xi32>
    // CHECK:   %[[c0:.*]] = arith.constant 0 : i32
    // CHECK:   linalg.fill ins(%[[c0]] : i32) outs(%[[res]] : memref<8xi32>)
    // CHECK:   cinm.op.gemv %{{.*}}, %{{.*}} into %[[res]] : memref<8x1024xi32>, memref<1024xi32> into memref<8xi32>
    // CHECK:   cinm.yield %[[res]] : memref<8xi32>
    // CHECK: }
        %r0 = cinm.compute attributes { workgroupShape=array<i64: 1, 8, 1> } -> tensor<8xi32> {
            %r = cinm.op.gemv %A, %B : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
            cinm.yield %r : tensor<8xi32>
        }
        func.return %r0 : tensor<8xi32>
    }

