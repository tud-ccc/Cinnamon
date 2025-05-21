#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %buf = empty_buf() : <f32>
    fill_buf %buf, 0xFF800000 : f32 : <f32>
    %buf_0 = empty_buf() : <(P), f32>
    fill_buf %buf_0, 0xFF800000 : f32 : <(P), f32>
    schedule<(red N) = (P * Q) to (1048576)>
            ins(%buf_3 = %arg0 : <(N), f32> to host)
            outs(%buf_4 = %buf_0 : <(P), f32>) {
      tile hwparallel factor symbolic<P> ins(%tile = %buf_3 sdim 0 : <(P * Q), f32>) outs(%tile_5 = %buf_4 sdim 0 : <(P), f32> rankreduce) {
        tile factor symbolic<Q> ins(%tile_6 = %tile sdim 0 : <(Q), f32>) {
          kernel "max" ins(%arg1 = %tile_6 : <1xf32>) outs(%arg2 = %tile_5 : <f32>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
    }
    schedule<(red N) = (P) to (P)>
            ins(%buf_3 = %buf_0 : <(N), f32>)
            outs(%buf_4 = %buf : <f32>) {
      tile reduction factor symbolic<P> ins(%tile = %buf_3 sdim 0 : <(P), f32>) {
        kernel "max" ins(%arg1 = %tile : <1xf32>) outs(%arg2 = %buf_4 : <f32>) {
          %0 = affine.load %arg1[0] : memref<1xf32>
          %1 = affine.load %arg2[] : memref<f32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %arg2[] : memref<f32>
        }
      }
    }
    %buf_1 = empty_buf() : <f32>
    fill_buf %buf_1, 0.000000e+00 : f32 : <f32>
    %buf_2 = empty_buf() : <(P1), f32>
    fill_buf %buf_2, 0.000000e+00 : f32 : <(P1), f32>
    schedule<(red M) = (P1 * Q) to (1048576)>
            ins(%buf_3 = %buf : <f32>)
            outs(%buf_4 = %arg0 : <(M), f32> to host, %buf_5 = %buf_2 : <(P1), f32>) {
      tile hwparallel factor symbolic<P1> outs(%tile = %buf_4 sdim 0 : <(P1 * Q), f32>, %tile_6 = %buf_5 sdim 0 : <(P1), f32> rankreduce) {
        tile factor symbolic<Q> outs(%tile_7 = %tile sdim 0 : <(Q), f32>) {
          kernel "sub+exp+sum" ins(%arg1 = %buf_3 : <f32>) outs(%arg2 = %tile_7 : <1xf32>, %arg3 = %tile_6 : <f32>) {
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
    schedule<(red N) = (P1) to (P1)>
            ins(%buf_3 = %buf_2 : <(N), f32>)
            outs(%buf_4 = %buf_1 : <f32>) {
      tile reduction factor symbolic<P1> ins(%tile = %buf_3 sdim 0 : <(P1), f32>) {
        kernel "sum" ins(%arg1 = %tile : <1xf32>) outs(%arg2 = %buf_4 : <f32>) {
          %0 = affine.load %arg1[0] : memref<1xf32>
          %1 = affine.load %arg2[] : memref<f32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg2[] : memref<f32>
        }
      }
    }
    schedule<(par M) = (P3 * Q) to (1048576)>
            ins(%buf_3 = %buf_1 : <f32>)
            outs(%buf_4 = %arg0 : <(M), f32> to host) {
      tile hwparallel factor symbolic<P3> outs(%tile = %buf_4 sdim 0 : <(P3 * Q), f32>) {
        tile parallel factor symbolic<Q> outs(%tile_5 = %tile sdim 0 : <(Q), f32>) {
          kernel "div" ins(%arg1 = %buf_3 : <f32>) outs(%arg2 = %tile_5 : <1xf32>) {
            %0 = affine.load %arg2[0] : memref<1xf32>
            %1 = affine.load %arg1[] : memref<f32>
            %2 = arith.divf %0, %1 : f32
            affine.store %2, %arg2[0] : memref<1xf32>
          }
        }
      }
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      // transform.btfl.expose_parallelism %arg1
      // %0 = transform.btfl.fuse_greedy %arg1 : (!transform.op<"btfl.block">) -> !transform.any_op
    }
  }
}

