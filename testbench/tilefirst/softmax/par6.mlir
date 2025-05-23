#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R in 1 to 8), dpus(d : D in 1 to 64), tasklets(t : T in 1 to 16)>
    %buf = empty_buf() : <f32, host>
    fill_buf %buf, 0xFF800000 : f32 : <f32, host>
    %buf_0 = empty_buf() : <(R * D), f32, host>
    fill_buf %buf_0, 0xFF800000 : f32 : <(R * D), f32, host>
    tile #upmem.launch<r * d> factor symbolic<R * D> ins(%tile = %arg0 sdim 0 : <1048576xf32, host>) outs(%tile_2 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce) {
      %buf_3 = empty_buf() : <(T), f32, wram(r,d)>
      fill_buf %buf_3, 0xFF800000 : f32 : <(T), f32, wram(r,d)>
      schedule<(red N) = (T * Q) to (1048576 | (R * D))>
              ins(%buf_4 = %tile : <(N), f32, wram(r, d)> to host)
              outs(%buf_5 = %buf_3 : <(T), f32, wram(r, d)>) {
        tile #threads.each factor symbolic<T> ins(%tile_6 = %buf_4 sdim 0 : <(T * Q), f32, wram(r, d)>) outs(%tile_7 = %buf_5 sdim 0 : <(T), f32, wram(r, d)> rankreduce) {
          tile factor symbolic<Q> ins(%tile_8 = %tile_6 sdim 0 : <(Q), f32, wram(r, d)>) {
            kernel "max" ins(%arg1 = %tile_8 : <1xf32, wram(r, d)>) outs(%arg2 = %tile_7 : <f32, wram(r, d)>) {
              %0 = affine.load %arg1[0] : memref<1xf32>
              %1 = affine.load %arg2[] : memref<f32>
              %2 = arith.maximumf %0, %1 : f32
              affine.store %2, %arg2[] : memref<f32>
            }
          }
        }
      }
      schedule<() = () to ()>
              ins(%buf_4 = %buf_3 : <(T), f32, wram(r, d)>)
              outs(%buf_5 = %tile_2 : <f32, wram(r, d)> to host) {
        tile reduction factor symbolic<T> ins(%tile_6 = %buf_4 sdim 0 : <(T), f32, wram(r, d)>) {
          kernel "max" ins(%arg1 = %tile_6 : <1xf32, wram(r, d)>) outs(%arg2 = %buf_5 : <f32, wram(r, d)>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
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
    tile #upmem.launch<r * d> factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_2 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce) {
      %buf_3 = empty_buf() : <(T), f32, wram(r, d)>
      fill_buf %buf_3, 0.000000e+00 : f32 : <(T), f32, wram(r, d)>
      schedule<(red M) = (T * Q) to (1048576 | (R * D))>
              ins(%buf_4 = %buf : <f32, wram(r, d)> to host)
              outs(%buf_5 = %tile : <(M), f32, wram(r, d)> to host, %buf_6 = %buf_3 : <(T), f32, wram(r, d)>) {
        tile #threads.each factor symbolic<T> outs(%tile_7 = %buf_5 sdim 0 : <(T * Q), f32, wram(r, d)>, %tile_8 = %buf_6 sdim 0 : <(T), f32, wram(r, d)> rankreduce) {
          tile factor symbolic<Q> outs(%tile_9 = %tile_7 sdim 0 : <(Q), f32, wram(r, d)>) {
            kernel "sub+exp+sum" ins(%arg1 = %buf_4 : <f32, wram(r, d)>) outs(%arg2 = %tile_9 : <1xf32, wram(r, d)>, %arg3 = %tile_8 : <f32, wram(r, d)>) {
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
      }
      schedule<() = () to ()>
              ins(%buf_4 = %buf_3 : <(T), f32, wram(r, d)>)
              outs(%buf_5 = %tile_2 : <f32, wram(r, d)> to host) {
        tile reduction factor symbolic<T> ins(%tile_6 = %buf_4 sdim 0 : <(T), f32, wram(r, d)>) {
          kernel "sum" ins(%arg1 = %tile_6 : <1xf32, wram(r, d)>) outs(%arg2 = %buf_5 : <f32, wram(r, d)>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
    }
    tile reduction factor symbolic<R * D> ins(%tile = %buf_0 sdim 0 : <(R * D), f32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_1 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile #upmem.launch<r * d> factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>) {
      schedule<(par M) = (T * Q) to (1048576 | (R * D))>
              ins(%buf_2 = %buf_1 : <f32, wram(r, d)> to host)
              outs(%buf_3 = %tile : <(M), f32, wram(r, d)> to host) {
        tile #threads.each factor symbolic<T> outs(%tile_4 = %buf_3 sdim 0 : <(T * Q), f32, wram(r, d)>) {
          tile parallel factor symbolic<Q> outs(%tile_5 = %tile_4 sdim 0 : <(Q), f32, wram(r, d)>) {
            kernel "div" ins(%arg1 = %buf_2 : <f32, wram(r, d)>) outs(%arg2 = %tile_5 : <1xf32, wram(r, d)>) {
              %0 = affine.load %arg2[0] : memref<1xf32>
              %1 = affine.load %arg1[] : memref<f32>
              %2 = arith.divf %0, %1 : f32
              affine.store %2, %arg2[0] : memref<1xf32>
            }
          }
        }
      }
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.block_solve_greedy %arg1 variables [T, Q]
    }
  }
}

