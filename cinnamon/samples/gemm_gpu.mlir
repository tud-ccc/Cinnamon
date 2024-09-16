#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d1 mod 128)>
#map2 = affine_map<(d0, d1) -> (d1 floordiv 128)>
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
          %c0_6 = arith.constant 0 : index
          scf.for %arg6 = %c0 to %c128 step %c1 {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %4 = memref.load %subview_5[%arg7, %arg6] : memref<32x128xi32, strided<[1024, 1], offset: ?>>
              memref.store %4, %alloc_3[%arg6, %arg7] : memref<128x32xi32>
            }
          }
          %memref = gpu.alloc  () : memref<1x128x32xi32>
          %memref_7 = gpu.alloc  () : memref<1x128x32xi32>
          %memref_8 = gpu.alloc  () : memref<1x128xi32>
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 128 {
              %4 = affine.apply #map(%arg6, %arg7)
              %subview_12 = memref.subview %subview_4[%4, 0] [1, 32] [1, 1] : memref<1x32xi32, strided<[1024, 1], offset: ?>> to memref<32xi32, strided<[1], offset: ?>>
              %subview_13 = memref.subview %memref[%arg6, %arg7, 0] [1, 1, 32] [1, 1, 1] : memref<1x128x32xi32> to memref<32xi32, strided<[1], offset: ?>>
              memref.copy %subview_12, %subview_13 : memref<32xi32, strided<[1], offset: ?>> to memref<32xi32, strided<[1], offset: ?>>
            }
          }
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 128 {
              %4 = affine.apply #map1(%arg6, %arg7)
              %subview_12 = memref.subview %alloc_3[%4, 0] [1, 32] [1, 1] : memref<128x32xi32> to memref<32xi32, strided<[1], offset: ?>>
              %subview_13 = memref.subview %memref_7[%arg6, %arg7, 0] [1, 1, 32] [1, 1, 1] : memref<1x128x32xi32> to memref<32xi32, strided<[1], offset: ?>>
              memref.copy %subview_12, %subview_13 : memref<32xi32, strided<[1], offset: ?>> to memref<32xi32, strided<[1], offset: ?>>
            }
          }
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 128 {
              %4 = affine.apply #map2(%arg6, %arg7)
              %5 = affine.apply #map1(%arg6, %arg7)
              %subview_12 = memref.subview %arg5[%4, %5] [1, 1] [1, 1] : memref<1x128xi32> to memref<i32, strided<[], offset: ?>>
              %subview_13 = memref.subview %memref_8[%arg6, %arg7] [1, 1] [1, 1] : memref<1x128xi32> to memref<i32, strided<[], offset: ?>>
              memref.copy %subview_12, %subview_13 : memref<i32, strided<[], offset: ?>> to memref<i32, strided<[], offset: ?>>
            }
          }
          %c1_9 = arith.constant 1 : index
          %c1_10 = arith.constant 1 : index
          %c128_11 = arith.constant 128 : index
          gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %c1_10, %arg13 = %c128_11, %arg14 = %c1_9) threads(%arg9, %arg10, %arg11) in (%arg15 = %c1_9, %arg16 = %c1_9, %arg17 = %c1_9) {
            %subview_12 = memref.subview %memref[%arg6, %arg7, 0] [1, 1, 32] [1, 1, 1] : memref<1x128x32xi32> to memref<32xi32, strided<[1], offset: ?>>
            %subview_13 = memref.subview %memref_7[%arg6, %arg7, 0] [1, 1, 32] [1, 1, 1] : memref<1x128x32xi32> to memref<32xi32, strided<[1], offset: ?>>
            %subview_14 = memref.subview %memref_8[%arg6, %arg7] [1, 1] [1, 1] : memref<1x128xi32> to memref<i32, strided<[], offset: ?>>
            %c0_15 = arith.constant 0 : index
            %c32_16 = arith.constant 32 : index
            %c1_17 = arith.constant 1 : index
            scf.for %arg18 = %c0_15 to %c32_16 step %c1_17 {
              %4 = memref.load %subview_12[%arg18] : memref<32xi32, strided<[1], offset: ?>>
              %5 = memref.load %subview_13[%arg18] : memref<32xi32, strided<[1], offset: ?>>
              %6 = memref.load %subview_14[] : memref<i32, strided<[], offset: ?>>
              %7 = arith.muli %4, %5 : i32
              %8 = arith.addi %7, %6 : i32
              memref.store %8, %subview_14[] : memref<i32, strided<[], offset: ?>>
            }
            gpu.terminator
          }
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 128 {
              %4 = affine.apply #map2(%arg6, %arg7)
              %5 = affine.apply #map1(%arg6, %arg7)
              %subview_12 = memref.subview %memref_8[%arg6, %arg7] [1, 1] [1, 1] : memref<1x128xi32> to memref<i32, strided<[], offset: ?>>
              %subview_13 = memref.subview %arg5[%4, %5] [1, 1] [1, 1] : memref<1x128xi32> to memref<i32, strided<[], offset: ?>>
              memref.copy %subview_12, %subview_13 : memref<i32, strided<[], offset: ?>> to memref<i32, strided<[], offset: ?>>
            }
          }
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

