#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d1 mod 128)>
#map2 = affine_map<(d0, d1) -> (d1 floordiv 128, d1 mod 128)>
module {
  memref.global "private" constant @__constant_1x128xi32 : memref<1x128xi32> = dense<0> {alignment = 64 : i64}
  func.func @main() {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x128xi32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128x32xi32>
    %0 = scf.for %arg0 = %c0 to %c1024 step %c1 iter_args(%arg1 = %alloc_1) -> (memref<1024x1024xi32>) {
      %1 = scf.for %arg2 = %c0 to %c1024 step %c128 iter_args(%arg3 = %arg1) -> (memref<1024x1024xi32>) {
        %2 = memref.get_global @__constant_1x128xi32 : memref<1x128xi32>
        memref.copy %2, %alloc_2 : memref<1x128xi32> to memref<1x128xi32>
        %3 = scf.for %arg4 = %c0 to %c1024 step %c32 iter_args(%arg5 = %alloc_2) -> (memref<1x128xi32>) {
          %subview_4 = memref.subview %alloc[%arg0, %arg4] [1, 32] [1, 1] : memref<1024x1024xi32> to memref<1x32xi32, strided<[1024, 1], offset: ?>>
          %subview_5 = memref.subview %alloc_0[%arg4, %arg2] [32, 128] [1, 1] : memref<1024x1024xi32> to memref<32x128xi32, strided<[1024, 1], offset: ?>>
          %4 = cnm.workgroup : !cnm.workgroup<1x128>
          scf.for %arg6 = %c0 to %c128 step %c1 {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %8 = memref.load %subview_5[%arg7, %arg6] : memref<32x128xi32, strided<[1024, 1], offset: ?>>
              memref.store %8, %alloc_3[%arg6, %arg7] : memref<128x32xi32>
            }
          }
          %5 = cnm.alloc() for %4 : !cnm.buffer<32xi32 on 1x128, level 0>
          %6 = cnm.alloc() for %4 : !cnm.buffer<32xi32 on 1x128, level 0>
          %7 = cnm.alloc() for %4 : !cnm.buffer<i32 on 1x128, level 0>
          cnm.scatter %subview_4 into %5[#map] of %4 : memref<1x32xi32, strided<[1024, 1], offset: ?>> into !cnm.buffer<32xi32 on 1x128, level 0>
          cnm.scatter %alloc_3 into %6[#map1] of %4 : memref<128x32xi32> into !cnm.buffer<32xi32 on 1x128, level 0>
          cnm.scatter %arg5 into %7[#map2] of %4 : memref<1x128xi32> into !cnm.buffer<i32 on 1x128, level 0>
          cnm.launch %4 in(%5, %6 : !cnm.buffer<32xi32 on 1x128, level 0>, !cnm.buffer<32xi32 on 1x128, level 0>) out(%7 : !cnm.buffer<i32 on 1x128, level 0>) on !cnm.workgroup<1x128> {
          ^bb0(%arg6: memref<32xi32>, %arg7: memref<32xi32>, %arg8: memref<i32>):
            %c0_6 = arith.constant 0 : index
            %c32_7 = arith.constant 32 : index
            %c1_8 = arith.constant 1 : index
            scf.for %arg9 = %c0_6 to %c32_7 step %c1_8 {
              %8 = memref.load %arg6[%arg9] : memref<32xi32>
              %9 = memref.load %arg7[%arg9] : memref<32xi32>
              %10 = memref.load %arg8[] : memref<i32>
              %11 = arith.muli %8, %9 : i32
              %12 = arith.addi %11, %10 : i32
              memref.store %12, %arg8[] : memref<i32>
            }
          }
          cnm.gather %7[#map2] of %4 into %arg5 : !cnm.buffer<i32 on 1x128, level 0> into memref<1x128xi32>
          scf.yield %arg5 : memref<1x128xi32>
        }
        %subview = memref.subview %arg3[%arg0, %arg2] [1, 128] [1, 1] : memref<1024x1024xi32> to memref<1x128xi32, strided<[1024, 1], offset: ?>>
        memref.copy %3, %subview : memref<1x128xi32> to memref<1x128xi32, strided<[1024, 1], offset: ?>>
        scf.yield %arg3 : memref<1024x1024xi32>
      }
      scf.yield %1 : memref<1024x1024xi32>
    }
    return
  }
}
