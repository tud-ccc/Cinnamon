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


// -----
#upmem = #upmem.platform<type = v1A, dimensions = 40x64x24>

module {
  func.func @simplify_iter_args(%arg0: tensor<768x768xf32>, %arg1: tensor<768xf32>) -> tensor<768xf32> {
    %0 = cinm.compute on accelerator #upmem.array<1x16x8, <type = v1A, dimensions = 40x64x24>> -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
      %1 = tensor.empty() : tensor<768xf32>
      %2 = affine.for %i = 0 to 768 step 256 iter_args(%acc = %1) -> (tensor<768xf32>) {
        %cst = arith.constant dense<0.000000e+00> : tensor<256xf32>
        %inserted_slice = tensor.insert_slice %cst into %acc[%i] [256] [1] : tensor<256xf32> into tensor<768xf32>
        %3 = affine.for %i_0 = 0 to 768 step 16 iter_args(%acc_1 = %inserted_slice) -> (tensor<768xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%i, %i_0] [256, 16] [1, 1] : tensor<768x768xf32> to tensor<256x16xf32>
          %extracted_slice_2 = tensor.extract_slice %arg1[%i_0] [16] [1] : tensor<768xf32> to tensor<16xf32>
          %extracted_slice_3 = tensor.extract_slice %acc_1[%i] [256] [1] : tensor<768xf32> to tensor<256xf32>
          %4 = cinm.op.gemv %extracted_slice, %extracted_slice_2 plus %extracted_slice_3 into %extracted_slice_3 : tensor<256x16xf32>, tensor<16xf32> plus tensor<256xf32> into tensor<256xf32> -> tensor<256xf32>
          %inserted_slice_4 = tensor.insert_slice %4 into %acc_1[%i] [256] [1] : tensor<256xf32> into tensor<768xf32>
          affine.yield %inserted_slice_4 : tensor<768xf32>
        }
        affine.yield %3 : tensor<768xf32>
      }
      cinm.yield %2 : tensor<768xf32>
    }
    return %0 : tensor<768xf32>
  }
}