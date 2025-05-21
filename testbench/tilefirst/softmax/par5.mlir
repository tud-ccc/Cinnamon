#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %buf = empty_buf() : <f32, host>
    fill_buf %buf, 0xFF800000 : f32 : <f32, host>
    %buf_0 = empty_buf() : <(R * D), f32, host>
    fill_buf %buf_0, 0xFF800000 : f32 : <(R * D), f32, host>
    tile[r * d] hwparallel factor symbolic<R * D> ins(%tile = %arg0 sdim 0 : <1048576xf32, host>) outs(%tile_3 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce) {
      schedule<(red N) = (1) to (1048576 | (R * D))>
              ins(%buf_4 = %tile : <(N), f32> to host)
              outs(%buf_5 = %tile_3 : <f32> to host) {
        kernel "max" ins(%arg1 = %buf_4 : <1xf32>) outs(%arg2 = %buf_5 : <f32>) {
          %0 = affine.load %arg1[0] : memref<1xf32>
          %1 = affine.load %arg2[] : memref<f32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %arg2[] : memref<f32>
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
    %buf_2 = empty_buf() : <(R * D), f32, host>
    fill_buf %buf_2, 0.000000e+00 : f32 : <(R * D), f32, host>
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, %tile_3 = %buf_2 sdim 0 : <(R * D), f32, host> rankreduce) {
      schedule<(par M) = (1) to (1048576 | (R * D))>
              ins(%buf_4 = %buf : <f32> to host)
              outs(%buf_5 = %tile : <(M), f32> to host) {
        kernel "sub+exp" ins(%arg1 = %buf_4 : <f32>) outs(%arg2 = %buf_5 : <1xf32>) {
          %0 = affine.load %arg2[0] : memref<1xf32>
          %1 = affine.load %arg1[] : memref<f32>
          %2 = arith.subf %0, %1 : f32
          %3 = math.exp %2 : f32
          affine.store %3, %arg2[0] : memref<1xf32>
        }
      }
      schedule<(red M) = (1) to (1048576 | (R * D))>
              ins(%buf_5 = %tile : <(M), f32> to host)
              outs(%buf_6 = %tile_3 : <f32> to host) {
        kernel "sum" ins(%arg2 = %buf_5 : <1xf32>) outs(%arg3 = %buf_6 : <f32>) {
          %0 = affine.load %arg2[0] : memref<1xf32>
          %4 = affine.load %arg3[] : memref<f32>
          %5 = arith.addf %0, %4 : f32
          affine.store %5, %arg3[] : memref<f32>
        }
      }
    }
    tile reduction factor symbolic<R * D> ins(%tile = %buf_2 sdim 0 : <(R * D), f32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_1 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>) {
      schedule<(par M) = (1) to (1048576 | (R * D))>
              ins(%buf_3 = %buf_1 : <f32> to host)
              outs(%buf_4 = %tile : <(M), f32> to host) {
        kernel "div" ins(%arg1 = %buf_3 : <f32>) outs(%arg2 = %buf_4 : <1xf32>) {
          %0 = affine.load %arg2[0] : memref<1xf32>
          %1 = affine.load %arg1[] : memref<f32>
          %2 = arith.divf %0, %1 : f32
          affine.store %2, %arg2[0] : memref<1xf32>
        }
      }
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.expose_parallelism %arg1
    }
  }
}

