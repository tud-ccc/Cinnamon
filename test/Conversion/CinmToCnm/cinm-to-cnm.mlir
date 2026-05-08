// RUN: cinm-opt --split-input-file --convert-cinm-to-cnm --canonicalize %s | FileCheck %s

#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<8x128x1, #upmem_platform>

// CHECK-LABEL: mm_dimm8_nopt
    func.func @mm_dimm8_nopt(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024x128xi32>) -> tensor<8x128xi32> {

// CHECK: %[[cst0:.*]] = arith.constant dense<0> : tensor<8x128xi32>
// CHECK: %[[wg:.*]] = cnm.workgroup : !cnm.workgroup<{{.*}}>
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<128x1024xi32>
// CHECK: %[[transposed:.*]] = linalg.transpose ins(%arg1 : tensor<1024x128xi32>) outs(%[[empty]] : tensor<128x1024xi32>) permutation = [1, 0]
// CHECK: %[[ba:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bb:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bc:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.scatter %arg0 into %[[ba]][{{.*}}] of %[[wg]] : tensor<8x1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %[[transposed]] into %[[bb]][{{.*}}] of %[[wg]] : tensor<128x1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %[[cst0]] into %[[bc]][{{.*}}] of %[[wg]] : tensor<8x128xi32> into !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.launch %[[wg]] ins(%{{.*}} = %[[ba]] : <1024xi32>, %{{.*}} = %[[bb]] : <1024xi32>) outs(%{{.*}} = %[[bc]] : <i32>) on {{.*}} {
// CHECK:    linalg.contract
// CHECK: %[[emptyres:.*]] = tensor.empty() : tensor<8x128xi32>
// CHECK: %{{.*}} = cnm.gather %[[bc]][{{.*}}] of %[[wg]] into %[[emptyres]] : !cnm.buffer<i32 on {{.*}}> into tensor<8x128xi32>
// CHECK: cnm.free_workgroup %[[wg]] : !cnm.workgroup<{{.*}}>
        %r0 = cinm.compute on accelerator #upmem -> tensor<8x128xi32> {
            %r = cinm.op.gemm %arg0, %arg1: tensor<8x1024xi32>, tensor<1024x128xi32> -> tensor<8x128xi32>
            cinm.yield %r : tensor<8x128xi32>
        }
        func.return %r0 : tensor<8x128xi32>
    }

// -----
// CHECK-LABEL: @gemv
#upmem_platform = #upmem.platform<type=v1A, dimensions = 4x16>
#upmem = #upmem.array<2x4x1, #upmem_platform>

    func.func @gemv(%arg0: tensor<8x1024xi32>, %arg1: tensor<1024xi32>) -> tensor<8xi32> {

// CHECK: %[[cst0:.*]] = arith.constant dense<0> : tensor<8xi32>
// CHECK: %[[wg:.*]] = cnm.workgroup : !cnm.workgroup<{{.*}}>
// CHECK: %[[ba:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %arg0 into %[[ba]][{{.*}}] of %[[wg]] : tensor<8x1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bb:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: cnm.scatter %arg1 into %[[bb]][{{.*}}] of %[[wg]] : tensor<1024xi32> into !cnm.buffer<1024xi32 on {{.*}}>
// CHECK: %[[bc:.*]] = cnm.alloc() for %[[wg]] : !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.scatter %[[cst0]] into %[[bc]][{{.*}}] of %[[wg]] : tensor<8xi32> into !cnm.buffer<i32 on {{.*}}>
// CHECK: cnm.launch %[[wg]] ins(%{{.*}} = %[[ba]] : <1024xi32>, %{{.*}} = %[[bb]] : <1024xi32>) outs(%{{.*}} = %[[bc]] : <i32>) on {{.*}} {
// CHECK:    linalg.contract
// CHECK: %[[emptyres:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %{{.*}} = cnm.gather %[[bc]][{{.*}}] of %[[wg]] into %[[emptyres]] : !cnm.buffer<i32 on {{.*}}> into tensor<8xi32>
// CHECK: cnm.free_workgroup %[[wg]] : !cnm.workgroup<{{.*}}>
        %r0 = cinm.compute on accelerator #upmem -> tensor<8xi32>  {
            %r = cinm.op.gemv %arg0, %arg1 : tensor<8x1024xi32>, tensor<1024xi32> -> tensor<8xi32>
            cinm.yield %r : tensor<8xi32>
        }
        func.return %r0 : tensor<8xi32>
    }
