#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R in 1 to 8), dpus(d : D in 1 to 64), tasklets(t : T = 8)>
    %buf = empty_buf() : <(R * D), f32, host>
    %buf_0 = empty_buf() : <1048576xf32, mram(r, d)>
    %buf_1 = empty_buf() : <(R * D), f32, mram(r, d)>
    %buf_2 = empty_buf() : <(R * D), f32, mram(r, d)>
    tile[r * d] hwparallel factor symbolic<R * D> ins(%tile = %arg0 sdim 0 : <1048576xf32, host>) outs(%tile_5 = %buf sdim 0 : <(R * D), f32, host> rankreduce, %tile_6 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_7 = %buf_1 sdim 0 : <(R * D), f32, mram(r, d)> rankreduce) {
      transfer %tile into %tile_6 : <(1048576 | (R * D)), f32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_6 : <(1048576 | (R * D)), f32, mram(r, d)>) outs(%arg2 = %tile_7 : <f32, mram(r, d)>) {
        %buf_8 = empty_buf() : <8xf32, wram(r, d)>
        threads.on_thread 0 {
          btfl.fill_buf %buf_8, 0xFF800000 : f32 : <8xf32, wram(r, d)>
        }
        threads.barrier
        tile factor symbolic<128 | (R * D)> ins(%tile_10 = %arg1 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_11 = empty_buf() : <8192xf32, wram(r, d)>
          threads.on_thread 0 {
            btfl.transfer %tile_10 into %buf_11 : <8192xf32, mram(r, d)> to wram(r, d)
          }
          threads.barrier
          tile #threads.each factor 8 ins(%tile_12 = %buf_11 sdim 0 : <8192xf32, wram(r, d)>) outs(%tile_13 = %buf_8 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 ins(%tile_14 = %tile_12 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "max" ins(%arg3 = %tile_14 : <1xf32, wram(r, d)>) outs(%arg4 = %tile_13 : <f32, wram(r, d)>) {
                %0 = affine.load %arg3[0] : memref<1xf32>
                %1 = affine.load %arg4[] : memref<f32>
                %2 = arith.maximumf %0, %1 : f32
                affine.store %2, %arg4[] : memref<f32>
              }
            }
          }
        }
        %buf_9 = empty_buf() : <f32, wram(r, d)>
        threads.on_thread 0 {
          btfl.fill_buf %arg2, 0xFF800000 : f32 : <f32, mram(r, d)>
        }
        threads.barrier
        threads.on_thread 0 {
          btfl.tile reduction factor 8 ins(%tile_10 = %buf_8 sdim 0 : <8xf32, wram(r, d)>) {
            kernel "max" ins(%arg3 = %tile_10 : <1xf32, wram(r, d)>) outs(%arg4 = %buf_9 : <f32, wram(r, d)>) {
              %0 = affine.load %arg3[0] : memref<1xf32>
              %1 = affine.load %arg4[] : memref<f32>
              %2 = arith.maximumf %0, %1 : f32
              affine.store %2, %arg4[] : memref<f32>
            }
          }
        }
        threads.on_thread 0 {
          btfl.transfer %buf_9 into %arg2 : <f32, wram(r, d)> to mram(r, d)
        }
      }
      transfer %tile_7 into %tile_5 : <f32, mram(r, d)> to host
    }
    %buf_3 = empty_buf() : <f32, host>
    fill_buf %buf_3, 0xFF800000 : f32 : <f32, host>
    tile reduction factor symbolic<R * D> ins(%tile = %buf sdim 0 : <(R * D), f32, host>) {
      kernel "max" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_3 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.maximumf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_5 = %buf sdim 0 : <(R * D), f32, host> rankreduce, %tile_6 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_7 = %buf_1 sdim 0 : <(R * D), f32, mram(r, d)> rankreduce, %tile_8 = %buf_2 sdim 0 : <(R * D), f32, mram(r, d)> rankreduce) {
      transfer %buf_3 into %tile_7 : <f32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_7 : <f32, mram(r, d)>) outs(%arg2 = %tile_6 : <(1048576 | (R * D)), f32, mram(r, d)>, %arg3 = %tile_8 : <f32, mram(r, d)>) {
        %buf_9 = empty_buf() : <8xf32, wram(r, d)>
        threads.on_thread 0 {
          btfl.fill_buf %buf_9, 0.000000e+00 : f32 : <8xf32, wram(r, d)>
        }
        threads.barrier
        tile factor symbolic<128 | (R * D)> outs(%tile_11 = %arg2 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_12 = empty_buf() : <f32, wram(r, d)>
          %buf_13 = empty_buf() : <8192xf32, wram(r, d)>
          threads.on_thread 0 {
            btfl.transfer %arg1 into %buf_12 : <f32, mram(r, d)> to wram(r, d)
          }
          threads.on_thread 1 {
            btfl.transfer %tile_11 into %buf_13 : <8192xf32, mram(r, d)> to wram(r, d)
          }
          threads.barrier
          tile #threads.each factor 8 outs(%tile_14 = %buf_13 sdim 0 : <8192xf32, wram(r, d)>, %tile_15 = %buf_9 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 outs(%tile_16 = %tile_14 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "sub+exp+sum" ins(%arg4 = %buf_12 : <f32, wram(r, d)>) outs(%arg5 = %tile_16 : <1xf32, wram(r, d)>, %arg6 = %tile_15 : <f32, wram(r, d)>) {
                %0 = affine.load %arg5[0] : memref<1xf32>
                %1 = affine.load %arg4[] : memref<f32>
                %2 = arith.subf %0, %1 : f32
                %3 = math.exp %2 : f32
                affine.store %3, %arg5[0] : memref<1xf32>
                %4 = affine.load %arg6[] : memref<f32>
                %5 = arith.addf %3, %4 : f32
                affine.store %5, %arg6[] : memref<f32>
              }
            }
          }
          threads.barrier
          threads.on_thread 0 {
            btfl.transfer %buf_13 into %tile_11 : <8192xf32, wram(r, d)> to mram(r, d)
          }
        }
        %buf_10 = empty_buf() : <f32, wram(r, d)>
        threads.on_thread 1 {
          btfl.fill_buf %arg3, 0.000000e+00 : f32 : <f32, mram(r, d)>
        }
        threads.on_thread 2 {
          btfl.tile reduction factor 8 ins(%tile_11 = %buf_9 sdim 0 : <8xf32, wram(r, d)>) {
            kernel "sum" ins(%arg4 = %tile_11 : <1xf32, wram(r, d)>) outs(%arg5 = %buf_10 : <f32, wram(r, d)>) {
              %0 = affine.load %arg4[0] : memref<1xf32>
              %1 = affine.load %arg5[] : memref<f32>
              %2 = arith.addf %0, %1 : f32
              affine.store %2, %arg5[] : memref<f32>
            }
          }
        }
        threads.barrier
        threads.on_thread 0 {
          btfl.transfer %buf_10 into %arg3 : <f32, wram(r, d)> to mram(r, d)
        }
      }
      transfer %tile_8 into %tile_5 : <f32, mram(r, d)> to host
    }
    %buf_4 = empty_buf() : <f32, host>
    fill_buf %buf_4, 0.000000e+00 : f32 : <f32, host>
    tile reduction factor symbolic<R * D> ins(%tile = %buf sdim 0 : <(R * D), f32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_4 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_5 = %buf_0 sdim 0 : <1048576xf32, mram(r, d)>, %tile_6 = %buf_2 sdim 0 : <(R * D), f32, mram(r, d)> rankreduce) {
      transfer %buf_4 into %tile_6 : <f32, host> to mram(r, d)
      scope(%upmem) attributes {threads.count = 8 : i32} ins(%arg1 = %tile_6 : <f32, mram(r, d)>) outs(%arg2 = %tile_5 : <(1048576 | (R * D)), f32, mram(r, d)>) {
        tile parallel factor symbolic<128 | (R * D)> outs(%tile_7 = %arg2 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_8 = empty_buf() : <f32, wram(r, d)>
          %buf_9 = empty_buf() : <8192xf32, wram(r, d)>
          threads.on_thread 0 {
            btfl.transfer %arg1 into %buf_8 : <f32, mram(r, d)> to wram(r, d)
          }
          threads.on_thread 1 {
            btfl.transfer %tile_7 into %buf_9 : <8192xf32, mram(r, d)> to wram(r, d)
          }
          threads.barrier
          tile #threads.each factor 8 outs(%tile_10 = %buf_9 sdim 0 : <8192xf32, wram(r, d)>) {
            tile parallel factor 1024 outs(%tile_11 = %tile_10 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "div" ins(%arg3 = %buf_8 : <f32, wram(r, d)>) outs(%arg4 = %tile_11 : <1xf32, wram(r, d)>) {
                %0 = affine.load %arg4[0] : memref<1xf32>
                %1 = affine.load %arg3[] : memref<f32>
                %2 = arith.divf %0, %1 : f32
                affine.store %2, %arg4[0] : memref<1xf32>
              }
            }
          }
          threads.barrier
          threads.on_thread 0 {
            btfl.transfer %buf_9 into %tile_7 : <8192xf32, wram(r, d)> to mram(r, d)
          }
        }
      }
      transfer %tile_5 into %tile : <(1048576 | (R * D)), f32, mram(r, d)> to host
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.interpret_variables %arg1 variables [R, D] = [2, 32]
      transform.btfl.lower.create_scoped_kernels %arg1
      transform.btfl.lower.finish_accelerator_lowering %arg1
    }
  }
}

