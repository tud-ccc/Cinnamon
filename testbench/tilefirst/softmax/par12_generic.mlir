#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R = 2), dpus(d : D = 32), tasklets(t : T = 8)>
    %buf = empty_buf() : <64xf32, host>
    %buf_0 = empty_buf() : <1048576xf32, mram(r, d)>
    %buf_1 = empty_buf() : <64xf32, mram(r, d)>
    %buf_2 = empty_buf() : <64xf32, mram(r, d)>
    tile[r * d] hwparallel factor 64 ins(%tile = %arg0 sdim 0 : <1048576xf32, host>) outs(%tile_5 = %buf sdim 0 : <64xf32, host> rankreduce, %tile_6 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_7 = %buf_1 sdim 0 : <64xf32, mram(r, d)> rankreduce) {
      transfer %tile into %tile_6 : <16384xf32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_6 : <16384xf32, mram(r, d)>) outs(%arg2 = %tile_7 : <f32, mram(r, d)>) {
        kernel "max+max" ins(%arg3 = %arg1 : <16384xf32, mram(r, d)> as memref<16384xf32, "mram">) outs(%arg4 = %arg2 : <f32, mram(r, d)> as memref<f32, "mram">) {
          %cst = arith.constant 0xFF800000 : f32
          %c8192 = arith.constant 8192 : index
          %wram_buf = upmem.static_alloc(wram) : memref<8xf32, "wram">
          threads.on_thread 0 {
            affine.for %arg5 = 0 to 8 {
              affine.store %cst, %wram_buf[%arg5] : memref<8xf32, "wram">
            }
          }
          threads.barrier
          %0 = threads.get_thread_id
          %1 = affine.load %wram_buf[symbol(%0)] : memref<8xf32, "wram">
          %2 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %1) -> (f32) {
            %3 = arith.muli %arg5, %c8192 : index
            %subview = memref.subview %arg3[%3] [8192] [1] : memref<16384xf32, "mram"> to memref<8192xf32, strided<[1], offset: ?>, "mram">
            %wram_buf_9 = upmem.static_alloc(wram) : memref<8192xf32, "wram">
            threads.on_thread 0 {
              upmem.local_transfer %subview into %wram_buf_9 : memref<8192xf32, strided<[1], offset: ?>, "mram"> to memref<8192xf32, "wram">
            }
            threads.barrier
            %4 = affine.for %arg7 = 0 to 1024 iter_args(%arg8 = %arg6) -> (f32) {
              %5 = affine.load %wram_buf_9[%arg7 + symbol(%0) * 1024] : memref<8192xf32, "wram">
              %6 = arith.maximumf %5, %arg8 : f32
              affine.yield %6 : f32
            }
            affine.yield %4 : f32
          }
          affine.store %2, %wram_buf[symbol(%0)] : memref<8xf32, "wram">
          %wram_buf_8 = upmem.static_alloc(wram) : memref<f32, "wram">
          threads.on_thread 0 {
            affine.store %cst, %arg4[] : memref<f32, "mram">
          }
          threads.barrier
          threads.on_thread 0 {
            %3 = affine.load %wram_buf_8[] : memref<f32, "wram">
            %4 = affine.for %arg5 = 0 to 8 iter_args(%arg6 = %3) -> (f32) {
              %5 = affine.load %wram_buf[%arg5] : memref<8xf32, "wram">
              %6 = arith.maximumf %5, %arg6 : f32
              affine.yield %6 : f32
            }
            affine.store %4, %wram_buf_8[] : memref<f32, "wram">
          }
          threads.on_thread 0 {
            upmem.local_transfer %wram_buf_8 into %arg4 : memref<f32, "wram"> to memref<f32, "mram">
          }
        }
      }
      transfer %tile_7 into %tile_5 : <f32, mram(r, d)> to host
    }
    %buf_3 = empty_buf() : <f32, host>
    fill_buf %buf_3, 0xFF800000 : f32 : <f32, host>
    tile reduction factor 64 ins(%tile = %buf sdim 0 : <64xf32, host>) {
      kernel "max" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_3 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.maximumf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor 64 outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_5 = %buf sdim 0 : <64xf32, host> rankreduce, %tile_6 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_7 = %buf_1 sdim 0 : <64xf32, mram(r, d)> rankreduce, %tile_8 = %buf_2 sdim 0 : <64xf32, mram(r, d)> rankreduce) {
      transfer %buf_3 into %tile_7 : <f32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_7 : <f32, mram(r, d)>) outs(%arg2 = %tile_6 : <16384xf32, mram(r, d)>, %arg3 = %tile_8 : <f32, mram(r, d)>) {
        kernel "sub+exp+sum+sum" ins(%arg4 = %arg1 : <f32, mram(r, d)> as memref<f32, "mram">) outs(%arg5 = %arg2 : <16384xf32, mram(r, d)> as memref<16384xf32, "mram">, %arg6 = %arg3 : <f32, mram(r, d)> as memref<f32, "mram">) {
          %cst = arith.constant 0.000000e+00 : f32
          %c8192 = arith.constant 8192 : index
          %wram_buf = upmem.static_alloc(wram) : memref<8xf32, "wram">
          threads.on_thread 0 {
            affine.for %arg7 = 0 to 8 {
              affine.store %cst, %wram_buf[%arg7] : memref<8xf32, "wram">
            }
          }
          threads.barrier
          %0 = threads.get_thread_id
          affine.for %arg7 = 0 to 2 {
            %1 = arith.muli %arg7, %c8192 : index
            %subview = memref.subview %arg5[%1] [8192] [1] : memref<16384xf32, "mram"> to memref<8192xf32, strided<[1], offset: ?>, "mram">
            %wram_buf_10 = upmem.static_alloc(wram) : memref<8192xf32, "wram">
            %wram_buf_11 = upmem.static_alloc(wram) : memref<f32, "wram">
            threads.on_thread 0 {
              upmem.local_transfer %arg4 into %wram_buf_11 : memref<f32, "mram"> to memref<f32, "wram">
            }
            threads.on_thread 1 {
              upmem.local_transfer %subview into %wram_buf_10 : memref<8192xf32, strided<[1], offset: ?>, "mram"> to memref<8192xf32, "wram">
            }
            threads.barrier
            %2 = affine.load %wram_buf[symbol(%0)] : memref<8xf32, "wram">
            %3 = affine.load %wram_buf_11[] : memref<f32, "wram">
            %4 = affine.for %arg8 = 0 to 1024 iter_args(%arg9 = %2) -> (f32) {
              %5 = affine.load %wram_buf_10[%arg8 + symbol(%0) * 1024] : memref<8192xf32, "wram">
              %6 = arith.subf %5, %3 : f32
              %7 = math.exp %6 : f32
              affine.store %7, %wram_buf_10[%arg8 + symbol(%0) * 1024] : memref<8192xf32, "wram">
              %8 = arith.addf %7, %arg9 : f32
              affine.yield %8 : f32
            }
            affine.store %4, %wram_buf[symbol(%0)] : memref<8xf32, "wram">
            threads.barrier
            threads.on_thread 0 {
              upmem.local_transfer %wram_buf_10 into %subview : memref<8192xf32, "wram"> to memref<8192xf32, strided<[1], offset: ?>, "mram">
            }
          }
          %wram_buf_9 = upmem.static_alloc(wram) : memref<f32, "wram">
          threads.on_thread 1 {
            affine.store %cst, %arg6[] : memref<f32, "mram">
          }
          threads.on_thread 2 {
            %1 = affine.load %wram_buf_9[] : memref<f32, "wram">
            %2 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %1) -> (f32) {
              %3 = affine.load %wram_buf[%arg7] : memref<8xf32, "wram">
              %4 = arith.addf %3, %arg8 : f32
              affine.yield %4 : f32
            }
            affine.store %2, %wram_buf_9[] : memref<f32, "wram">
          }
          threads.barrier
          threads.on_thread 0 {
            upmem.local_transfer %wram_buf_9 into %arg6 : memref<f32, "wram"> to memref<f32, "mram">
          }
        }
      }
      transfer %tile_8 into %tile_5 : <f32, mram(r, d)> to host
    }
    %buf_4 = empty_buf() : <f32, host>
    fill_buf %buf_4, 0.000000e+00 : f32 : <f32, host>
    tile reduction factor 64 ins(%tile = %buf sdim 0 : <64xf32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_4 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor 64 outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_5 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_6 = %buf_2 sdim 0 : <64xf32, mram(r, d)> rankreduce) {
      transfer %buf_4 into %tile_6 : <f32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_6 : <f32, mram(r, d)>) outs(%arg2 = %tile_5 : <16384xf32, mram(r, d)>) {
        kernel "div" ins(%arg3 = %arg1 : <f32, mram(r, d)> as memref<f32, "mram">) outs(%arg4 = %arg2 : <16384xf32, mram(r, d)> as memref<16384xf32, "mram">) {
          %c8192 = arith.constant 8192 : index
          %0 = threads.get_thread_id
          affine.for %arg5 = 0 to 2 {
            %1 = arith.muli %arg5, %c8192 : index
            %subview = memref.subview %arg4[%1] [8192] [1] : memref<16384xf32, "mram"> to memref<8192xf32, strided<[1], offset: ?>, "mram">
            %wram_buf = upmem.static_alloc(wram) : memref<8192xf32, "wram">
            %wram_buf_7 = upmem.static_alloc(wram) : memref<f32, "wram">
            threads.on_thread 0 {
              upmem.local_transfer %arg3 into %wram_buf_7 : memref<f32, "mram"> to memref<f32, "wram">
            }
            threads.on_thread 1 {
              upmem.local_transfer %subview into %wram_buf : memref<8192xf32, strided<[1], offset: ?>, "mram"> to memref<8192xf32, "wram">
            }
            threads.barrier
            %2 = affine.load %wram_buf_7[] : memref<f32, "wram">
            affine.for %arg6 = 0 to 1024 {
              %3 = affine.load %wram_buf[%arg6 + symbol(%0) * 1024] : memref<8192xf32, "wram">
              %4 = arith.divf %3, %2 : f32
              affine.store %4, %wram_buf[%arg6 + symbol(%0) * 1024] : memref<8192xf32, "wram">
            }
            threads.barrier
            threads.on_thread 0 {
              upmem.local_transfer %wram_buf into %subview : memref<8192xf32, "wram"> to memref<8192xf32, strided<[1], offset: ?>, "mram">
            }
          }
        }
      }
      transfer %tile_5 into %tile : <16384xf32, mram(r, d)> to host
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.lower.finish_accelerator_lowering %arg1
    }
  }
}

