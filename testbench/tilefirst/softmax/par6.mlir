#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R1 = 1), dpus(d : D1 = 1), tasklets(t : T = 8)>
    %buf = empty_buf() : <f32, host>
    fill_buf %buf, 0xFF800000 : f32 : <f32, host>
    %buf_0 = empty_buf() : <(R * D), f32, host>
    fill_buf %buf_0, 0xFF800000 : f32 : <(R * D), f32, host>
    tile[r * d] hwparallel factor symbolic<R * D> ins(%tile = %arg0 sdim 0 : <1048576xf32, host>) outs(%tile_3 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce) {
      %buf_4 = empty_buf() : <8xf32, host>
      fill_buf %buf_4, 0xFF800000 : f32 : <8xf32, host>
      %buf_5 = empty_buf() : <(1048576 | (R * D)), f32, mram(r, d)>
      %buf_6 = empty_buf() : <8xf32, mram(r, d)>
      transfer %tile into %buf_5 : <(1048576 | (R * D)), f32, host> to mram(r, d)
      fill_buf %buf_6, 0xFF800000 : f32 : <8xf32, mram(r, d)>
      schedule<(red N) = (1048576) to (1048576 | (R * D))>
              ins(%buf_9 = %buf_5 : <(N), f32, mram(r, d)>)
              outs(%buf_10 = %buf_6 : <8xf32, mram(r, d)>) {
        tile factor 128 ins(%tile_11 = %buf_9 sdim 0 : <1048576xf32, mram(r, d)>) {
          %buf_12 = empty_buf() : <8192xf32, wram(r, d)>
          %buf_13 = empty_buf() : <8xf32, wram(r, d)>
          tile parallel factor 16 ins(%tile_14 = %tile_11 sdim 0 : <8192xf32, mram(r, d)>) outs(%tile_15 = %buf_12 sdim 0 : <8192xf32, wram(r, d)>) {
            transfer %tile_14 into %tile_15 : <512xf32, mram(r, d)> to wram(r, d)
          }
          transfer %buf_10 into %buf_13 : <8xf32, mram(r, d)> to wram(r, d)
          tile #threads.each factor 8 ins(%tile_14 = %buf_12 sdim 0 : <8192xf32, wram(r, d)>) outs(%tile_15 = %buf_13 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 ins(%tile_16 = %tile_14 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "max" ins(%arg1 = %tile_16 : <1xf32, wram(r, d)>) outs(%arg2 = %tile_15 : <f32, wram(r, d)>) {
                %0 = affine.load %arg1[0] : memref<1xf32>
                %1 = affine.load %arg2[] : memref<f32>
                %2 = arith.maximumf %0, %1 : f32
                affine.store %2, %arg2[] : memref<f32>
              }
            }
          }
          transfer %buf_13 into %buf_10 : <8xf32, wram(r, d)> to mram(r, d)
        }
      }
      %buf_7 = empty_buf() : <8xf32, mram(r, d)>
      %buf_8 = empty_buf() : <f32, mram(r, d)>
      transfer %buf_6 into %buf_4 : <8xf32, mram(r, d)> to host
      transfer %buf_4 into %buf_7 : <8xf32, host> to mram(r, d)
      fill_buf %buf_8, 0xFF800000 : f32 : <f32, mram(r, d)>
      schedule<() = () to ()>
              ins(%buf_9 = %buf_7 : <8xf32, host> to mram(r, d))
              outs(%buf_10 = %buf_8 : <f32, host> to mram(r, d)) {
        tile reduction factor 8 ins(%tile_11 = %buf_9 sdim 0 : <8xf32, host>) {
          kernel "max" ins(%arg1 = %tile_11 : <1xf32, host>) outs(%arg2 = %buf_10 : <f32, host>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
      transfer %buf_8 into %tile_3 : <f32, mram(r, d)> to host
    }
    tile reduction factor symbolic<R * D> ins(%tile = %buf_0 sdim 0 : <(R * D), f32, host>) {
      kernel "max" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.maximumf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    %buf_1 = empty_buf() : <f32, host>
    fill_buf %buf_1, 0.000000e+00 : f32 : <f32, host>
    fill_buf %buf_0, 0.000000e+00 : f32 : <(R * D), f32, host>
    %buf_2 = empty_buf() : <f32, mram(r, d)>
    transfer %buf into %buf_2 : <f32, host> to mram(r, d)
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_3 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce) {
      %buf_4 = empty_buf() : <8xf32, host>
      fill_buf %buf_4, 0.000000e+00 : f32 : <8xf32, host>
      %buf_5 = empty_buf() : <(1048576 | (R * D)), f32, mram(r, d)>
      %buf_6 = empty_buf() : <8xf32, mram(r, d)>
      transfer %tile into %buf_5 : <(1048576 | (R * D)), f32, host> to mram(r, d)
      fill_buf %buf_6, 0.000000e+00 : f32 : <8xf32, mram(r, d)>
      schedule<(red M) = (1048576) to (1048576 | (R * D))>
              ins(%buf_9 = %buf_2 : <f32, mram(r, d)>)
              outs(%buf_10 = %buf_5 : <(M), f32, mram(r, d)>, %buf_11 = %buf_6 : <8xf32, mram(r, d)>) {
        tile factor 128 outs(%tile_12 = %buf_10 sdim 0 : <1048576xf32, mram(r, d)>) {
          %buf_13 = empty_buf() : <f32, wram(r, d)>
          %buf_14 = empty_buf() : <8192xf32, wram(r, d)>
          %buf_15 = empty_buf() : <8xf32, wram(r, d)>
          transfer %buf_9 into %buf_13 : <f32, mram(r, d)> to wram(r, d)
          tile parallel factor 16 ins(%tile_16 = %tile_12 sdim 0 : <8192xf32, mram(r, d)>) outs(%tile_17 = %buf_14 sdim 0 : <8192xf32, wram(r, d)>) {
            transfer %tile_16 into %tile_17 : <512xf32, mram(r, d)> to wram(r, d)
          }
          transfer %buf_11 into %buf_15 : <8xf32, mram(r, d)> to wram(r, d)
          tile #threads.each factor 8 outs(%tile_16 = %buf_14 sdim 0 : <8192xf32, wram(r, d)>, %tile_17 = %buf_15 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 outs(%tile_18 = %tile_16 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "sub+exp+sum" ins(%arg1 = %buf_13 : <f32, wram(r, d)>) outs(%arg2 = %tile_18 : <1xf32, wram(r, d)>, %arg3 = %tile_17 : <f32, wram(r, d)>) {
                %0 = affine.load %arg2[0] : memref<1xf32>
                %1 = affine.load %arg1[] : memref<f32>
                %2 = arith.subf %0, %1 : f32
                %3 = math.exp %2 : f32
                affine.store %3, %arg2[0] : memref<1xf32>
                %4 = affine.load %arg3[] : memref<f32>
                %5 = arith.addf %3, %4 : f32
                affine.store %5, %arg3[] : memref<f32>
              }
            }
          }
          tile parallel factor 16 ins(%tile_16 = %buf_14 sdim 0 : <8192xf32, wram(r, d)>) outs(%tile_17 = %tile_12 sdim 0 : <8192xf32, mram(r, d)>) {
            transfer %tile_16 into %tile_17 : <512xf32, wram(r, d)> to mram(r, d)
          }
          transfer %buf_15 into %buf_11 : <8xf32, wram(r, d)> to mram(r, d)
        }
      }
      %buf_7 = empty_buf() : <8xf32, mram(r, d)>
      %buf_8 = empty_buf() : <f32, mram(r, d)>
      transfer %buf_5 into %tile : <(1048576 | (R * D)), f32, mram(r, d)> to host
      transfer %buf_6 into %buf_4 : <8xf32, mram(r, d)> to host
      transfer %buf_4 into %buf_7 : <8xf32, host> to mram(r, d)
      fill_buf %buf_8, 0.000000e+00 : f32 : <f32, mram(r, d)>
      schedule<() = () to ()>
              ins(%buf_9 = %buf_7 : <8xf32, host> to mram(r, d))
              outs(%buf_10 = %buf_8 : <f32, host> to mram(r, d)) {
        tile reduction factor 8 ins(%tile_11 = %buf_9 sdim 0 : <8xf32, host>) {
          kernel "sum" ins(%arg1 = %tile_11 : <1xf32, host>) outs(%arg2 = %buf_10 : <f32, host>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
      transfer %buf_8 into %tile_3 : <f32, mram(r, d)> to host
    }
    tile reduction factor symbolic<R * D> ins(%tile = %buf_0 sdim 0 : <(R * D), f32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_1 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    transfer %buf_1 into %buf_2 : <f32, host> to mram(r, d)
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>) {
      %buf_3 = empty_buf() : <(1048576 | (R * D)), f32, mram(r, d)>
      transfer %tile into %buf_3 : <(1048576 | (R * D)), f32, host> to mram(r, d)
      schedule<(par M) = (1048576) to (1048576 | (R * D))>
              ins(%buf_4 = %buf_2 : <f32, mram(r, d)>)
              outs(%buf_5 = %buf_3 : <(M), f32, mram(r, d)>) {
        tile factor 128 outs(%tile_6 = %buf_5 sdim 0 : <1048576xf32, mram(r, d)>) {
          %buf_7 = empty_buf() : <f32, wram(r, d)>
          %buf_8 = empty_buf() : <8192xf32, wram(r, d)>
          transfer %buf_4 into %buf_7 : <f32, mram(r, d)> to wram(r, d)
          tile parallel factor 16 ins(%tile_9 = %tile_6 sdim 0 : <8192xf32, mram(r, d)>) outs(%tile_10 = %buf_8 sdim 0 : <8192xf32, wram(r, d)>) {
            transfer %tile_9 into %tile_10 : <512xf32, mram(r, d)> to wram(r, d)
          }
          tile #threads.each factor 8 outs(%tile_9 = %buf_8 sdim 0 : <8192xf32, wram(r, d)>) {
            tile parallel factor 1024 outs(%tile_10 = %tile_9 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "div" ins(%arg1 = %buf_7 : <f32, wram(r, d)>) outs(%arg2 = %tile_10 : <1xf32, wram(r, d)>) {
                %0 = affine.load %arg2[0] : memref<1xf32>
                %1 = affine.load %arg1[] : memref<f32>
                %2 = arith.divf %0, %1 : f32
                affine.store %2, %arg2[0] : memref<1xf32>
              }
            }
          }
          tile parallel factor 16 ins(%tile_9 = %buf_8 sdim 0 : <8192xf32, wram(r, d)>) outs(%tile_10 = %tile_6 sdim 0 : <8192xf32, mram(r, d)>) {
            transfer %tile_9 into %tile_10 : <512xf32, wram(r, d)> to mram(r, d)
          }
        }
      }
      transfer %buf_3 into %tile : <(1048576 | (R * D)), f32, mram(r, d)> to host
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.schedule_block %arg1
    }
  }
}

