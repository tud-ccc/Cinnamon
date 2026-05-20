// RUN: cinm-opt --cinm-tiling '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' %s --cse --canonicalize --cse --canonicalize --split-input-file | cinm-opt | FileCheck %s

// CHECK-LABEL: gemm
    func.func @gemm(%A: tensor<8x1024xi32>, %B: tensor<1024x128xi32>) -> tensor<8x128xi32> {

    // CHECK:       cinm.compute_block ({{.*}}) -> memref<8x128xi32>
    // CHECK:         %[[alloc:.*]] = memref.alloc() {{.*}} : memref<8x128xi32>
    // CHECK:           affine.for %[[J:.*]] = 0 to 128 step 32
    // CHECK:             memref.get_global @__constant_8x32xi32 : memref<8x32xi32>
    // CHECK:             %[[tile:.*]] = memref.subview %[[alloc]][0, %[[J]]] [8, 32] [1, 1]
    // CHECK:             memref.copy {{.*}}, %[[tile]]
    // CHECK:             affine.for %[[K:.*]] = 0 to 1024 step 128
    // CHECK:               cinm.op.gemm {{.*}} into %[[tile]]
    // CHECK:         cinm.yield %[[alloc]] : memref<8x128xi32>
        %r0 = cinm.compute_block (%a = %A : tensor<8x1024xi32>, %b = %B : tensor<1024x128xi32>) -> tensor<8x128xi32> {
            %r = cinm.op.gemm %a, %b {cinm.tile_sizes = array<i64: 8, 32, 128>} : tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }


// -----
// CHECK-LABEL: gemv
    func.func @gemv(%A: tensor<8x1024xi32>, %B: tensor<1024xi32>) -> tensor<8xi32> {

    // CHECK:       cinm.compute_block ({{.*}}) -> memref<8xi32>
    // CHECK:         %[[alloc:.*]] = memref.alloc() {{.*}} : memref<8xi32>
    // CHECK:         affine.for %[[I:.*]] = 0 to 8
    // CHECK:           memref.get_global @__constant_1xi32 : memref<1xi32>
    // CHECK:           %[[tile:.*]] = memref.subview %[[alloc]][%[[I]]] [1] [1]
    // CHECK:           memref.copy {{.*}}, %[[tile]]
    // CHECK:           affine.for %[[J:.*]] = 0 to 1024 step 8
    // CHECK:             cinm.op.gemv {{.*}} into %[[tile]]
    // CHECK:         cinm.yield %[[alloc]] : memref<8xi32>
        %r0 = cinm.compute_block (%a = %A : tensor<8x1024xi32>, %b = %B : tensor<1024xi32>) -> tensor<8xi32> attributes { workgroupShape=array<i64: 1, 8, 1> } {
            %r = cinm.op.gemv %a, %b {cinm.tile_sizes = array<i64: 1, 8>}: tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
            cinm.yield %r : tensor<8xi32>
        }
        func.return %r0 : tensor<8xi32>
    }

