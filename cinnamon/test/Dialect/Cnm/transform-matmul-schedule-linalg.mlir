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

    // this is naive matmul
    cnm.compute
       ins(%arg0[(i, j) -> (i)]: memref<1024x64xi32>, 
           %arg1t[(i, j) -> (j)]: memref<1024x64xi32>)
       outs(%res[(i, j) -> (i, j)]: memref<1024x1024xi32>)
       on hierarchy<1024x1024>
       do (%a1: memref<64xi32>, %b1: memref<64xi32>, %o: memref<i32>)  {
       // reduction (could be raised to linalg.reduce)
       linalg.generic #matmul_trait
          ins(%a1, %b1: memref<64xi32>, memref<64xi32>)
          outs(%o: memref<i32>) {
            ^bb0(%a: i32, %b: i32, %c: i32):
              %0 = arith.muli %a, %b: i32
              %1 = arith.addi %0, %c: i32
              linalg.yield %1: i32
          }
      }
    
    // expand dim 1 factor 64  -> <1024x64x16>
    // expand dim 0 factor 64  -> <64x16x64x16>
    // swap dim 1 and 2        -> <64x64x16x16>
    cnm.compute
      ins(%arg0[(d0, d1, d2, d3) -> (d0 * 16 + d2)] : memref<1024x64xi32>, 
          %arg1t[(d0, d1, d2, d3) -> (d1 * 16 + d3)] : memref<1024x64xi32>)
      outs(%res[(d0, d1, d2, d3) -> (d0 * 16 + d2, d1 * 16 + d3)] : memref<1024x1024xi32>) 
      on hierarchy<64x64x16x16>
      do (%a1: memref<64xi32>, %b1: memref<64xi32>, %o: memref<i32>) {
       linalg.generic #matmul_trait
          ins(%a1, %b1: memref<64xi32>, memref<64xi32>)
          outs(%o: memref<i32>) {
            ^bb0(%a: i32, %b: i32, %c: i32):
              %0 = arith.muli %a, %b: i32
              %1 = arith.addi %0, %c: i32
              linalg.yield %1: i32
          }
      }

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
         linalg.generic #matmul_trait
          ins(%arg2, %arg3: memref<64xi32>, memref<64xi32>)
          outs(%arg4: memref<i32>) {
            ^bb0(%a: i32, %b: i32, %c: i32):
              %0 = arith.muli %a, %b: i32
              %1 = arith.addi %0, %c: i32
              linalg.yield %1: i32
          }
        }
    }

    // Then, let's say we want to lift the reduction out.

    // Let's say that we want to tile the reduction loop with factor 2.
    // We need
    // - tiling factor
    // - reduction iterator index (just do one reduction at a time)
    // We want something like
    // Notice:
    // - Kernel argument types change but not the linalg generic (in particular, not affine maps)
    // - The reduction between the two iterations is done implicitly because
    // the output of the first iteration is used to initialize the second.
    // - We need to reshape inputs. Which ones? The ones that 
    //    - Are inputs of the linalg generic, and
    //    - Use the index of the targeted reduction iterator (look at affine maps), and
    // - Also we need the index space of the reduction iterator to be divisible by the factor
    // Simultaneously reshape 64 into 2x32, and introduce a surrounding loop, and introduce a symbol in the affine maps 
    // to index the new input dimension.
    //
    // A more stepwise lowering would be
    // - Tile the linalg generic, making a new reduction iterator appear as the outer iterator
    // - Make a transform that extracts an outermost red iterator
    // - Q, does it need to be outermost?
    //
    affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
      affine.for %k0 = 0 to 64 step 32 {
        %arg00 = memref.expand_shape %arg0 [[0], [1, 2]]: memref<1024x64xi32> into memref<1024x2x32xi32>
        %arg10 = memref.expand_shape %arg1t [[0], [1, 2]]: memref<1024x64xi32> into memref<1024x2x32xi32>
        
        // this does only half of the 64 reduction, returns a partial result
        cnm.compute
          symbols [%D0, %D1, %k0]
          ins(%arg00[(d2, d3)[D0, D1, K] -> (D0 * 16 + d2, K floordiv 32)] : memref<1024x2x32xi32>, 
              %arg10[(d2, d3)[D0, D1, K] -> (D1 * 16 + d3, K floordiv 32)] : memref<1024x2x32xi32>)
          outs(%res[(d2, d3)[D0, D1, K] -> (D0 * 16 + d2, D1 * 16 + d3)] : memref<1024x1024xi32>) 
          on hierarchy<16x16>
          do (%arg2: memref<32xi32>, %arg3: memref<32xi32>, %arg4: memref<i32>) {
            linalg.generic #matmul_trait
              ins(%arg2, %arg3: memref<32xi32>, memref<32xi32>)
              outs(%arg4: memref<i32>) {
                ^bb0(%a: i32, %b: i32, %c: i32):
                  %0 = arith.muli %a, %b: i32
                  %1 = arith.addi %0, %c: i32
                  linalg.yield %1: i32
              }
          }
      }
    }
    // Ok stepwise:
    affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
        // first tile the inner kernel
        // (todo do that automatically with transform. Is that possible?)
        cnm.compute
          symbols [%D0, %D1]
          ins(%arg0[(d2, d3)[D0, D1] -> (D0 * 16 + d2)] : memref<1024x64xi32>, 
              %arg1t[(d2, d3)[D0, D1] -> (D1 * 16 + d3)] : memref<1024x64xi32>)
          outs(%res[(d2, d3)[D0, D1] -> (D0 * 16 + d2, D1 * 16 + d3)] : memref<1024x1024xi32>) 
          on hierarchy<16x16>
          do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
            %arg20 = memref.expand_shape %arg2 [[0, 1]]: memref<64xi32> into memref<2x32xi32>
            %arg30 = memref.expand_shape %arg3 [[0, 1]]: memref<64xi32> into memref<2x32xi32>
            linalg.generic {
              indexing_maps = [
                affine_map<(ko, kt) -> (ko, kt)>,
                affine_map<(ko, kt) -> (ko, kt)>,
                affine_map<(ko, kt) -> ()>
              ],
              iterator_types = ["reduction", "reduction"]
              }
              ins(%arg20, %arg30: memref<2x32xi32>, memref<2x32xi32>)
              outs(%arg4: memref<i32>) {
                ^bb0(%a: i32, %b: i32, %c: i32):
                  %0 = arith.muli %a, %b: i32
                  %1 = arith.addi %0, %c: i32
                  linalg.yield %1: i32
              }
          }
      }

    // fork
    cnm.compute
      ins(%arg0[(d0, d1, d2, d3) -> (d0 * 16 + d2)] : memref<1024x64xi32>, 
          %arg1t[(d0, d1, d2, d3) -> (d1 * 16 + d3)] : memref<1024x64xi32>)
      outs(%res[(d0, d1, d2, d3) -> (d0 * 16 + d2, d1 * 16 + d3)] : memref<1024x1024xi32>) 
      on hierarchy<64x64x16x16>
      do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
        affine.for %arg5 = 0 to 64 {
          %0 = affine.load %arg2[%arg5] : memref<64xi32>
          %1 = affine.load %arg3[%arg5] : memref<64xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg4[] : memref<i32>
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %arg4[] : memref<i32>
        }
      }

    // peel right (need a reshape first), reshape turns output map into (i,j,k,l) -> (i*16+k,j,l)
    %exp0 = memref.expand_shape %res[[0], [1, 2]] : memref<1024x1024xi32> into memref<1024x64x16xi32>
    %expb = memref.expand_shape %arg1t[[0, 1], [2]] : memref<1024x64xi32> into memref<64x16x64xi32>

    cnm.compute
      ins(%arg0[(d0, d1, d2) -> (d0 * 16 + d2)] : memref<1024x64xi32>, 
          %expb[(d0, d1, d2) -> (d1)] : memref<64x16x64xi32>)
      outs(%exp0[(d0, d1, d2) -> (d0 * 16 + d2, d1)] : memref<1024x64x16xi32>) 
      on hierarchy<64x64x16>
      do (%arg2: memref<64xi32>, %arg3: memref<16x64xi32>, %arg4: memref<16xi32>) {
        affine.parallel (%x) = (0) to (16) {
          affine.for %arg5 = 0 to 64 {
            %0 = affine.load %arg2[%arg5] : memref<64xi32>
            %1 = affine.load %arg3[%x, %arg5] : memref<16x64xi32>
            %2 = arith.muli %0, %1 : i32
            %3 = affine.load %arg4[%x] : memref<16xi32>
            %4 = arith.addi %2, %3 : i32
            affine.store %4, %arg4[%x] : memref<16xi32>
          }
        }
      }



    // affine.parallel (%D0, %D1) = (0, 0) to (64, 64) {
    //   cnm.compute 
    //     symbols [%D0, %D1]
    //     ins(%arg0[(d2, d3)[D0, D1] -> (d3 * 64 + D1)] : memref<1024x64xi32>, 
    //         %arg1t[(d2, d3)[D0, D1] -> (d2 * 64 + D0)] : memref<1024x64xi32>)
    //     outs(%res[(d2, d3)[D0, D1] -> (d3 * 64 + D1, d2 * 64 + D0)] : memref<1024x1024xi32>) 
    //     on hierarchy<16x16>
    //     do (%arg2: memref<64xi32>, %arg3: memref<64xi32>, %arg4: memref<i32>) {
    //       affine.for %arg5 = 0 to 64 {
    //         %0 = affine.load %arg2[%arg5] : memref<64xi32>
    //         %1 = affine.load %arg2[%arg5] : memref<64xi32>
    //         %2 = arith.muli %0, %1 : i32
    //         %3 = affine.load %arg4[] : memref<i32>
    //         %4 = arith.addi %2, %3 : i32
    //         affine.store %4, %arg4[] : memref<i32>
    //       }
    //     }
    // }
    
    // into

    return
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"cnm.compute">):
    transform.cnm.expand_dim %arg0 dim 1 by factor 64: (!transform.op<"cnm.compute">) -> ()
    transform.cnm.expand_dim %arg0 dim 0 by factor 64: (!transform.op<"cnm.compute">) -> ()
    transform.cnm.swap_dims %arg0, 1, 2: (!transform.op<"cnm.compute">) -> ()
  }
}