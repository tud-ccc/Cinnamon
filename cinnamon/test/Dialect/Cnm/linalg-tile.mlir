// RUN: cinm-opt %s --cnm-apply-transform | FileCheck %s
#map = affine_map<(d0) -> (d0)>
#matmul_trait = {
  doc = "C() += A(k) * B(k)",
  indexing_maps = [
    affine_map<(k) -> (k)>,
    affine_map<(k) -> (k)>,
    affine_map<(k) -> ()>
  ],
  iterator_types = ["reduction"]
}

#linalg_trait = {

}
module {


   func.func @matmul(%arg0: memref<1024x64xi32>, %arg1: memref<64x1024xi32>) {
    %arg1t = memref.alloc() : memref<1024x64xi32>
    linalg.transpose ins(%arg1: memref<64x1024xi32>) outs(%arg1t: memref<1024x64xi32>) permutation = [1,0]
    %res = memref.alloc() : memref<1024x1024xi32>

    // peel left 2
    // This computes 16x16 tiles of the output on some dimms.
    // - Can we lift the reduction loop outside of the nest? That's hard
    // - Can we coarsen the output size (not use memref<i32> but memref<16xi32>)
    //   - That's fine.
    //     
    affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
      cnm.compute
        symbols [%D0, %D1]
        ins(%arg0[(d2, d3)[D0, D1] -> (D0 * 16 + d2)] : memref<1024x64xi32>, 
            %arg1t[(d2, d3)[D0, D1] -> (D1 * 16 + d3)] : memref<1024x64xi32>)
        outs(%res[(d2, d3)[D0, D1] -> (D0 * 16 + d2, D1 * 16 + d3)] : memref<1024x1024xi32>) 
        on hierarchy<16x16>
        do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
        %t0 = bufferization.to_tensor %arg2: memref<64xi32>
        %t1 = bufferization.to_tensor %arg3: memref<64xi32>
        %t3 = bufferization.to_tensor %arg4: memref<i32>
        %o = linalg.generic #matmul_trait
          ins(%t0, %t1: tensor<64xi32>, tensor<64xi32>)
          outs(%t3: tensor<i32>) {
            ^bb0(%a: i32, %b: i32, %c: i32):
              %0 = arith.muli %a, %b: i32
              %1 = arith.addi %0, %c: i32
              linalg.yield %1: i32
          } -> tensor<i32>
        }
    }


    // // Ok stepwise:
    // affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
    //     // first tile the inner kernel
    //     // (todo do that automatically with transform. Is that possible?)
    //     cnm.compute
    //       symbols [%D0, %D1]
    //       ins(%arg0[(d2, d3)[D0, D1] -> (D0 * 16 + d2)] : memref<1024x64xi32>, 
    //           %arg1t[(d2, d3)[D0, D1] -> (D1 * 16 + d3)] : memref<1024x64xi32>)
    //       outs(%res[(d2, d3)[D0, D1] -> (D0 * 16 + d2, D1 * 16 + d3)] : memref<1024x1024xi32>) 
    //       on hierarchy<16x16>
    //       do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
    //         %arg20 = memref.expand_shape %arg2 [[0, 1]]: memref<64xi32> into memref<2x32xi32>
    //         %arg30 = memref.expand_shape %arg3 [[0, 1]]: memref<64xi32> into memref<2x32xi32>
    //         linalg.generic {
    //           indexing_maps = [
    //             affine_map<(ko, kt) -> (ko, kt)>,
    //             affine_map<(ko, kt) -> (ko, kt)>,
    //             affine_map<(ko, kt) -> ()>
    //           ],
    //           iterator_types = ["reduction", "reduction"]
    //           }
    //           ins(%arg20, %arg30: memref<2x32xi32>, memref<2x32xi32>)
    //           outs(%arg4: memref<i32>) {
    //             ^bb0(%a: i32, %b: i32, %c: i32):
    //               %0 = arith.muli %a, %b: i32
    //               %1 = arith.addi %0, %c: i32
    //               linalg.yield %1: i32
    //           }
    //       }
    //   }

    return
  }
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"linalg.generic">):
    %r:4 = transform.structured.tile_reduction_using_for %arg0 
      by tile_sizes = [32]: (!transform.op<"linalg.generic">) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  }

  // transform.sequence failures(propagate) {
  // ^bb0(%arg0: !transform.op<"cnm.compute">):
  //   transform.cnm.expand_dim %arg0 dim 1 by factor 64: (!transform.op<"cnm.compute">) -> ()
  //   transform.cnm.expand_dim %arg0 dim 0 by factor 64: (!transform.op<"cnm.compute">) -> ()
  //   transform.cnm.swap_dims %arg0, 1, 2: (!transform.op<"cnm.compute">) -> ()
  // }
}