// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm --canonicalize %s | FileCheck %s

// With an accumulator wider than the operands, the two reduction operands and
// the accumulator are sized apart: a leaf holds K operand-typed elements of A
// and of B, and one accumulator-typed element of C. Getting the C buffer's
// element type from the operands instead would both scatter the initializer
// into a too-narrow buffer and accumulate the whole dot product in i8.

#upmem_platform = #upmem.platform<type = v1A, dpus = 1024, tasklets = 24>
#upmem = #upmem.array<1024x1, #upmem_platform>

// CHECK-LABEL: @gemm_i8_i32
func.func @gemm_i8_i32(%arg0: tensor<8x1024xi8>, %arg1: tensor<1024x128xi8>) -> tensor<8x128xi32> {
// CHECK: %[[cst0:.*]] = arith.constant dense<0> : tensor<8x128xi32>
// CHECK: %[[wg:.*]] = cnm.workgroup : !cnm.workgroup<{{.*}}>
// A and B keep the operand type ...
// CHECK: %[[ba:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<1024xi8 on {{.*}}>
// CHECK: %[[bb:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<1024xi8 on {{.*}}>
// ... and C the accumulator type.
// CHECK: %[[bc:.*]] = cnm.declare_buffer() for %[[wg]] : !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.scatter %arg0 into %[[ba]][{{.*}}] of %[[wg]] : tensor<8x1024xi8> into !cnm.buffer<1024xi8 on {{.*}}>
// CHECK: cnm.scatter %{{.*}} into %[[bb]][{{.*}}] of %[[wg]] : tensor<128x1024xi8> into !cnm.buffer<1024xi8 on {{.*}}>
// CHECK: cnm.scatter %[[cst0]] into %[[bc]][{{.*}}] of %[[wg]] : tensor<8x128xi32> into !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.launch %[[wg]] ins(%{{.*}} = %[[ba]] : <1024xi8>, %{{.*}} = %[[bb]] : <1024xi8>) outs(%{{.*}} = %[[bc]] : <i32>) on {{.*}} {
// CHECK:   linalg.contract {{.*}} ins(%{{.*}}, %{{.*}} : memref<1024xi8>, memref<1024xi8>) outs(%{{.*}} : memref<i32>)
// CHECK: cnm.gather %[[bc]][{{.*}}] of %[[wg]] into %{{.*}} : !cnm.buffer<i32 on {{.*}}> into tensor<8x128xi32>
    %r0 = cinm.compute on accelerator #upmem -> tensor<8x128xi32> {
        %r = cinm.op.gemm %arg0, %arg1 : tensor<8x1024xi8>, tensor<1024x128xi8> -> tensor<8x128xi32>
        cinm.yield %r : tensor<8x128xi32>
    }
    func.return %r0 : tensor<8x128xi32>
}

// -----

// gemv reaches the same place through convertCinmToCnm, which buffers each
// operand at its own element type already.

#upmem_platform = #upmem.platform<type = v1A, dpus = 64, tasklets = 24>
#upmem = #upmem.array<8x1, #upmem_platform>

// CHECK-LABEL: @gemv_i8_i32
func.func @gemv_i8_i32(%arg0: tensor<8x1024xi8>, %arg1: tensor<1024xi8>) -> tensor<8xi32> {
// CHECK: cnm.declare_buffer() for %{{.*}} : !cnm.buffer<1024xi8 on {{.*}}>
// CHECK: cnm.declare_buffer() for %{{.*}} : !cnm.buffer<1024xi8 on {{.*}}>
// CHECK: cnm.declare_buffer() for %{{.*}} : !cnm.buffer<i32 on {{.*}}>
// CHECK: linalg.contract {{.*}} ins(%{{.*}}, %{{.*}} : memref<1024xi8>, memref<1024xi8>) outs(%{{.*}} : memref<i32>)
    %r0 = cinm.compute on accelerator #upmem -> tensor<8xi32> {
        %r = cinm.op.gemv %arg0, %arg1 : tensor<8x1024xi8>, tensor<1024xi8> -> tensor<8xi32>
        cinm.yield %r : tensor<8xi32>
    }
    func.return %r0 : tensor<8xi32>
}
