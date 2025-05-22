#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R in 1 to 8), dpus(d : D in 1 to 64), tasklets(t : T = 8)>
    %buf_0 = empty_buf() : <(R * D), f32, host>

    // These are distributed buffers
    // They must be tiled by a 
    //    tile[r*d] hwparallel factor symbolic<R*D>
    %mirror = empty_buf() : <1048576xf32, mram(r, d)> 
    %partial_dist = empty_buf() : <(R * D), f32, mram(r, d)>
    %partial_dist_2 = empty_buf() : <(R * D), f32, mram(r, d)>

    tile[r * d] hwparallel factor symbolic<R * D> 
      ins(%tile = %arg0 sdim 0 : <1048576xf32, host>)
      outs(%tile_2 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce,
           %buf_4 = %mirror sdim 0 : <1048576xf32, mram(r, d)>,
           %buf_6 = %partial_dist sdim 0: <(R * D), f32, mram(r, d)> rankreduce) {

      transfer %tile into %buf_4 : <(1048576 | (R * D)), f32, host> to mram(r, d)

      scope(%upmem) attributes{threads.count=8} ins(%b4 = %buf_4 : <(1048576 | (R * D)), f32, mram(r, d)>)
                    outs(%b6 = %buf_6 : <f32, mram(r,d)>) {
        %buf_3 = empty_buf() : <8xf32, wram(r, d)>
        fill_buf %buf_3, 0xFF800000 : f32 : <8xf32, wram(r, d)>
        tile factor symbolic<128 | (R * D)> ins(%tile_7 = %b4 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_8 = empty_buf() : <8192xf32, wram(r, d)>
          transfer %tile_7 into %buf_8 : <8192xf32, mram(r, d)> to wram(r, d)
          tile #threads.each factor 8 ins(%tile_9 = %buf_8 sdim 0 : <8192xf32, wram(r, d)>) outs(%tile_10 = %buf_3 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 ins(%tile_11 = %tile_9 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "max" ins(%arg1 = %tile_11 : <1xf32, wram(r, d)>) outs(%arg2 = %tile_10 : <f32, wram(r, d)>) {
                %0 = affine.load %arg1[0] : memref<1xf32>
                %1 = affine.load %arg2[] : memref<f32>
                %2 = arith.maximumf %0, %1 : f32
                affine.store %2, %arg2[] : memref<f32>
              }
            }
          }
        }
        %buf_5 = empty_buf() : <f32, wram(r, d)>
        fill_buf %b6, 0xFF800000 : f32 : <f32, mram(r, d)>
        tile reduction factor 8 ins(%tile_7 = %buf_3 sdim 0 : <8xf32, wram(r, d)>) {
          kernel "max" ins(%arg1 = %tile_7 : <1xf32, wram(r, d)>) outs(%arg2 = %buf_5 : <f32, wram(r, d)>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
        transfer %buf_5 into %b6 : <f32, wram(r, d)> to mram(r, d)
      }
      transfer %buf_6 into %tile_2 : <f32, mram(r, d)> to host
    }
    %buf = empty_buf() : <f32, host>
    fill_buf %buf, 0xFF800000 : f32 : <f32, host>
    tile reduction factor symbolic<R * D> ins(%tile = %buf_0 sdim 0 : <(R * D), f32, host>) {
      kernel "max" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.maximumf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }

    tile[r * d] hwparallel factor symbolic<R * D> 
        outs(%tile = %arg0 sdim 0 : <1048576xf32, host>, 
             %tile_2 = %buf_0 sdim 0 : <(R * D), f32, host> rankreduce,
             %buf_4 = %mirror sdim 0 : <1048576xf32, mram(r, d)>,

             %buf_6 = %partial_dist sdim 0: <(R * D), f32, mram(r, d)> rankreduce,
             %buf_7 = %partial_dist_2 sdim 0: <(R * D), f32, mram(r, d)> rankreduce
             ) {

      transfer %buf into %buf_6 : <f32, host> to mram(r, d)
      scope(%upmem) attributes{threads.count=8}ins(%b44 = %buf_6 : <f32, mram(r,d)>)
                    outs(
                      %b55 = %buf_4 : <(1048576 | (R * D)), f32, mram(r, d)>,
                      %b77 = %buf_7 : <f32, mram(r,d)>) {
        %buf_3 = empty_buf() : <8xf32, wram(r, d)>
        fill_buf %buf_3, 0.000000e+00 : f32 : <8xf32, wram(r, d)>
        tile factor symbolic<128 | (R * D)> outs(%tile_8 = %b55 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_9 = empty_buf() : <f32, wram(r, d)>
          %buf_10 = empty_buf() : <8192xf32, wram(r, d)>
          transfer %b44 into %buf_9 : <f32, mram(r, d)> to wram(r, d)
          transfer %tile_8 into %buf_10 : <8192xf32, mram(r, d)> to wram(r, d)
          tile #threads.each factor 8 outs(%tile_11 = %buf_10 sdim 0 : <8192xf32, wram(r, d)>, %tile_12 = %buf_3 sdim 0 : <8xf32, wram(r, d)> rankreduce) {
            tile factor 1024 outs(%tile_13 = %tile_11 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "sub+exp+sum" ins(%arg1 = %buf_9 : <f32, wram(r, d)>) outs(%arg2 = %tile_13 : <1xf32, wram(r, d)>, %arg3 = %tile_12 : <f32, wram(r, d)>) {
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
          transfer %buf_10 into %tile_8 : <8192xf32, wram(r, d)> to mram(r, d)
        }
        %buf_6 = empty_buf() : <f32, wram(r, d)>
        fill_buf %b77, 0.000000e+00 : f32 : <f32, mram(r, d)>
        tile reduction factor 8 ins(%tile_8 = %buf_3 sdim 0 : <8xf32, wram(r, d)>) {
          kernel "sum" ins(%arg1 = %tile_8 : <1xf32, wram(r, d)>) outs(%arg2 = %buf_6 : <f32, wram(r, d)>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.addf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
        transfer %buf_6 into %b77 : <f32, wram(r, d)> to mram(r, d)
      }
      transfer %buf_7 into %tile_2 : <f32, mram(r, d)> to host
    }
    %buf_1 = empty_buf() : <f32, host>
    fill_buf %buf_1, 0.000000e+00 : f32 : <f32, host>
    tile reduction factor symbolic<R * D> ins(%tile = %buf_0 sdim 0 : <(R * D), f32, host>) {
      kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_1 : <f32, host>) {
        %0 = affine.load %arg1[0] : memref<1xf32>
        %1 = affine.load %arg2[] : memref<f32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[] : memref<f32>
      }
    }
    tile[r * d] hwparallel factor symbolic<R * D> outs(%tile = %arg0 sdim 0 : <1048576xf32, host>,
                                                       %buf_4 = %mirror sdim 0 : <1048576xf32, mram(r, d)>,
                                                       %buf_2 = %partial_dist_2 sdim 0 : <(R*D), f32, mram(r,d)> rankreduce) {
      transfer %buf_1 into %buf_2 : <f32, host> to mram(r, d)
      scope(%upmem) attributes{threads.count=8}ins(%b22 = %buf_2 : <f32, mram(r,d)>)
                    outs(%b33 = %buf_4 : <(1048576 | (R * D)), f32, mram(r, d)>) {
        tile parallel factor symbolic<128 | (R * D)> outs(%tile_4 = %b33 sdim 0 : <(1048576 | (R * D)), f32, mram(r, d)>) {
          %buf_5 = empty_buf() : <f32, wram(r, d)>
          %buf_6 = empty_buf() : <8192xf32, wram(r, d)>
          transfer %b22 into %buf_5 : <f32, mram(r, d)> to wram(r, d)
          transfer %tile_4 into %buf_6 : <8192xf32, mram(r, d)> to wram(r, d)
          tile #threads.each factor 8 outs(%tile_7 = %buf_6 sdim 0 : <8192xf32, wram(r, d)>) {
            tile parallel factor 1024 outs(%tile_8 = %tile_7 sdim 0 : <1024xf32, wram(r, d)>) {
              kernel "div" ins(%arg1 = %buf_5 : <f32, wram(r, d)>) outs(%arg2 = %tile_8 : <1xf32, wram(r, d)>) {
                %0 = affine.load %arg2[0] : memref<1xf32>
                %1 = affine.load %arg1[] : memref<f32>
                %2 = arith.divf %0, %1 : f32
                affine.store %2, %arg2[0] : memref<1xf32>
              }
            }
          }
          transfer %buf_6 into %tile_4 : <8192xf32, wram(r, d)> to mram(r, d)
        }
      }
      transfer %buf_4 into %tile : <(1048576 | (R * D)), f32, mram(r, d)> to host
    }
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.lower.prepare_thread_assignment %arg1
      transform.btfl.lower.do_threads_assignment %arg1
    }
  }
}

