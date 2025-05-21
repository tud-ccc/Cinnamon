#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <1048576xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R = 4), dpus(d : D = 64), tasklets(t : T = 16)>
    %buf = empty_buf() : <f32, host>
    fill_buf %buf, 0xFF800000 : f32 : <f32, host>
    %buf_0 = empty_buf() : <(P), f32, host>
    fill_buf %buf_0, 0xFF800000 : f32 : <(P), f32, host>
    schedule<(red N) = (P * Q) to (1048576)>
            ins(%buf_3 = %arg0 : <(N), f32, wram(r,d)> to host)
            outs(%buf_4 = %buf_0 : <(P), f32, wram(r,d)> to host) {
      tile hwparallel factor symbolic<P> ins(%tile = %buf_3 sdim 0 : <(P * Q), f32, wram(r,d)>) outs(%tile_5 = %buf_4 sdim 0 : <(P), f32, wram(r,d)> rankreduce) {
        tile factor symbolic<Q> ins(%tile_6 = %tile sdim 0 : <(Q), f32, wram(r,d)>) {
          kernel "max" ins(%arg1 = %tile_6 : <1xf32, wram(r,d)>) outs(%arg2 = %tile_5 : <f32, wram(r,d)>) {
            %0 = affine.load %arg1[0] : memref<1xf32>
            %1 = affine.load %arg2[] : memref<f32>
            %2 = arith.maximumf %0, %1 : f32
            affine.store %2, %arg2[] : memref<f32>
          }
        }
      }
    }
    schedule<(red N) = (P) to (P)>
            ins(%buf_3 = %buf_0 : <(N), f32, host>)
            outs(%buf_4 = %buf : <f32, host>) {
      tile reduction factor symbolic<P> ins(%tile = %buf_3 sdim 0 : <(P), f32, host>) {
        kernel "max" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_4 : <f32, host>) {
          %0 = affine.load %arg1[0] : memref<1xf32>
          %1 = affine.load %arg2[] : memref<f32>
          %2 = arith.maximumf %0, %1 : f32
          affine.store %2, %arg2[] : memref<f32>
        }
      }
    }
    %buf_1 = empty_buf() : <f32, host>
    fill_buf %buf_1, 0.000000e+00 : f32 : <f32, host>
    %buf_2 = empty_buf() : <(P1), f32, host>
    fill_buf %buf_2, 0.000000e+00 : f32 : <(P1), f32, host>
    schedule<(red M) = (P1 * Q) to (1048576)>
            ins(%buf_3 = %buf : <f32, wram(r,d)> to host)
            outs(%buf_4 = %arg0 : <(M), f32, wram(r,d)> to host, %buf_5 = %buf_2 : <(P1), f32, wram(r,d)> to host) {
      tile hwparallel factor symbolic<P1> outs(%tile = %buf_4 sdim 0 : <(P1 * Q), f32, wram(r,d)>, %tile_6 = %buf_5 sdim 0 : <(P1), f32, wram(r,d)> rankreduce) {
        tile factor symbolic<Q> outs(%tile_7 = %tile sdim 0 : <(Q), f32, wram(r,d)>) {
          kernel "sub+exp+sum" ins(%arg1 = %buf_3 : <f32, wram(r,d)>) outs(%arg2 = %tile_7 : <1xf32, wram(r,d)>, %arg3 = %tile_6 : <f32, wram(r,d)>) {
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
            ins(%buf_3 = %buf_2 : <(N), f32, host>)
            outs(%buf_4 = %buf_1 : <f32, host>) {
      tile reduction factor symbolic<P1> ins(%tile = %buf_3 sdim 0 : <(P1), f32, host>) {
        kernel "sum" ins(%arg1 = %tile : <1xf32, host>) outs(%arg2 = %buf_4 : <f32, host>) {
          %0 = affine.load %arg1[0] : memref<1xf32>
          %1 = affine.load %arg2[] : memref<f32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg2[] : memref<f32>
        }
      }
    }
    schedule<(par M) = (P3 * Q) to (1048576)>
            ins(%buf_3 = %buf_1 : <f32, wram(r,d)> to host)
            outs(%buf_4 = %arg0 : <(M), f32, wram(r,d)> to host) {
      tile hwparallel factor symbolic<P3> outs(%tile = %buf_4 sdim 0 : <(P3 * Q), f32, wram(r,d)>) {
        tile parallel factor symbolic<Q> outs(%tile_5 = %tile sdim 0 : <(Q), f32, wram(r,d)>) {
          kernel "div" ins(%arg1 = %buf_3 : <f32, wram(r,d)>) outs(%arg2 = %tile_5 : <1xf32, wram(r,d)>) {
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

      transform.btfl.interpret_variables %arg1 variables [P1, P3] = [P, P]
      transform.btfl.debug.snapshot_op_location %arg1 : !transform.op<"btfl.block">
      %schedules = transform.btfl.find_descendants "btfl.schedule" in %arg1 : (!transform.op<"btfl.block">) -> !transform.op<"btfl.schedule">
      transform.sequence %schedules : !transform.op<"btfl.schedule"> failures(suppress) {
      ^bb0(%arg2: !transform.op<"btfl.schedule">):
        // transform.btfl.interpret_variables %arg1 variables [P1, P3] = [P, P]
        // transform.btfl.debug.snapshot_op_location %arg1 : !transform.op<"btfl.block">
        // %schedules = transform.btfl.find_descendants "btfl.schedule" in %arg1 : (!transform.op<"btfl.block">) -> !transform.op<"btfl.schedule">
        transform.btfl.simplify_schedule %arg2 unwrap empty
        // transform.btfl.solve_greedy %schedules variables [P, Q] 
      }
      // transform.btfl.debug.snapshot_op_location %schedules : !transform.op<"btfl.schedule">
      // transform.btfl.simplify_schedule %schedules
      // transform.btfl.solve_greedy %schedules variables [P, Q] 
    }
  }
}

