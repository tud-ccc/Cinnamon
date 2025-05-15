
#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions=(8 x 64 x 16)>


module {

   btfl.block @softmax_initial outs(%O: <65536xf32, host>) platform #upmem {

    // %max = max(%A)
    %max = empty_buf() : <f32>
    fill_buf %max, 0xFF800000 : f32 : <f32>
    schedule<(red N)=(1) to (65536)> ins(%a = %O: <(N), f32, ?> to host) outs(%max1 = %max : <f32, ?> to ?) {
      kernel "max" ins(%x = %a : <1xf32, ?>) outs(%o = %max1: <f32, ?>) {
        %cst0 = arith.constant 0: index
        %x0 = memref.load %x[%cst0] : memref<1xf32>
        %o0 = memref.load %o[] : memref<f32>
        %o1 = arith.maximumf %x0, %o0 : f32
        memref.store %o1, %o[] : memref<f32>
      }
    }

    // %O = %A - splat(%max)
    schedule<(par M)=(1) to (65536)> ins(%m = %max: <f32, ?> to ?) outs(%o = %O : <(M), f32, ?> to host) {
      kernel "sub" ins(%m = %m: <f32, ?>) outs(%y = %o: <1xf32, ?>) {
        %cst0 = arith.constant 0: index
        %x0 = memref.load %y[%cst0] : memref<1xf32>
        %o0 = memref.load %m[] : memref<f32>
        %o1 = arith.subf %x0, %o0 : f32
        memref.store %o1, %y[%cst0] : memref<1xf32>
      }
    }

    // %O = exp(%O)
    schedule<(par M)=(1) to (65536)> outs(%o = %O : <(M), f32, ?> to host) {
      kernel "exp" outs(%y = %o: <1xf32, ?>) {
        %cst0 = arith.constant 0: index
        %x0 = memref.load %y[%cst0] : memref<1xf32>
        %o1 = math.exp %x0 : f32
        memref.store %o1, %y[%cst0] : memref<1xf32>
      }
    }

    // %sum = sum(%O)
    %sum = empty_buf() : <f32>
    fill_buf %sum, 0.0 : f32 : <f32>
    schedule<(red N)=(1) to (65536)> ins(%o = %O: <(N), f32, ?> to host) outs(%sum2 = %sum: <f32, ?> to ?) {
      kernel "sum" ins(%x = %o : <1xf32, ?>) outs(%ox = %sum2: <f32, ?>) {
        %cst0 = arith.constant 0: index
        %x0 = memref.load %x[%cst0] : memref<1xf32>
        %o0 = memref.load %ox[] : memref<f32>
        %o1 = arith.addf %x0, %o0 : f32
        memref.store %o1, %ox[] : memref<f32>
      }
    }
    

    // %O = %O / splat(%sum)
    schedule<(par M)=(1) to (65536)> ins(%sum2 = %sum: <f32, ?> to ?) outs(%o = %O : <(M), f32, ?> to host) {
      kernel "div" ins(%m = %sum2: <f32, ?>) outs(%y = %o: <1xf32, ?>) {
        %cst0 = arith.constant 0: index
        %y0 = memref.load %y[%cst0] : memref<1xf32>
        %sum2 = memref.load %m[] : memref<f32>
        %o1 = arith.divf %y0, %sum2 : f32
        memref.store %o1, %y[%cst0] : memref<1xf32>
      }
    }




    transform.sequence failures(propagate) {
      ^bb0(%arg1: !transform.op<"btfl.block">, %arg3: !transform.op<"btfl.schedule">):
        // transform.btfl.expose_parallelism %arg1
        // transform.btfl.fuse_greedy %arg1 : (!transform.op<"btfl.block">) -> !transform.any_op
        transform.btfl.schedule_block %arg1
        // // transform.btfl.buffer_reuse_analysis %arg1
        transform.btfl.lower.create_accelerator_scopes %arg1
        transform.btfl.lower.prepare_thread_assignment %arg1
        transform.btfl.lower.do_threads_assignment %arg1
        transform.btfl.lower.create_scoped_kernels %arg1
        // transform.btfl.fuse_greedy %arg1 : (!transform.op<"btfl.block">) -> !transform.any_op

        transform.yield
    }
  }
}