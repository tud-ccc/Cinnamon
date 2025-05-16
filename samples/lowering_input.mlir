#map = affine_map<(d0)[s0] -> (d0 * 8192 + s0 * 512)>
#mram = #tilefirst.level<name = "mram", size_in_bytes = 67108864, alignment = 8, arity = 2>
#wram = #tilefirst.level<name = "wram", size_in_bytes = 65536, alignment = 8, arity = 2>
#upmem = #upmem.platform<levels = [#mram, #wram], dimensions = (8 x 64 x 16)>
module {
  btfl.block @softmax_initial outs(%arg0 : <65536xf32, host>)  platform #upmem {
    %upmem = tilefirst.accelerator #upmem.array<ranks(r : R = 1), dpus(d : D = 1), tasklets(t : T = 16)>
    %buf = empty_buf() : <65536xf32, mram(r, d)>
    transfer %arg0 into %buf : <65536xf32, host> to mram(r, d)
    scope(%upmem) attributes {threads.count = 16 : i32} outs(%arg1 = %buf : <65536xf32, mram(r, d)>) {
      kernel "max+max+sub+exp+sum+sum+div" outs(%arg2 = %arg1 : <65536xf32, mram(r, d)> as memref<65536xf32, "mram">) {
        %wram_buf = upmem.static_alloc(wram) : memref<f32, "wram">
        %wram_buf_0 = upmem.static_alloc(wram) : memref<8192xf32, "wram">
        %wram_buf_1 = upmem.static_alloc(wram) : memref<16xf32, "wram">
        %wram_buf_2 = upmem.static_alloc(wram) : memref<f32, "wram">
        %cst = arith.constant 0.000000e+00 : f32
        %c512 = arith.constant 512 : index
        %cst_3 = arith.constant 0xFF800000 : f32
        threads.on_thread 0 {
          affine.store %cst_3, %wram_buf_2[] : memref<f32, "wram">
        }
        threads.on_thread 1 {
          affine.for %arg3 = 0 to 16 {
            affine.store %cst_3, %wram_buf_1[%arg3] : memref<16xf32, "wram">
          }
        }
        threads.barrier
        %0 = threads.get_thread_id
        %1 = arith.muli %0, %c512 : index
        %subview = memref.subview %wram_buf_0[%1] [512] [1] : memref<8192xf32, "wram"> to memref<512xf32, strided<[1], offset: ?>, "wram">
        %2 = affine.load %wram_buf_1[symbol(%0)] : memref<16xf32, "wram">
        %3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %2) -> (f32) {
          threads.barrier
          %6 = affine.apply #map(%arg3)[%0]
          %subview_4 = memref.subview %arg2[%6] [512] [1] : memref<65536xf32, "mram"> to memref<512xf32, strided<[1], offset: ?>, "mram">
          upmem.local_transfer %subview_4 into %subview : memref<512xf32, strided<[1], offset: ?>, "mram"> to memref<512xf32, strided<[1], offset: ?>, "wram">
          %7 = affine.for %arg5 = 0 to 512 iter_args(%arg6 = %arg4) -> (f32) {
            %8 = affine.load %wram_buf_0[%arg5 + symbol(%0) * 512] : memref<8192xf32, "wram">
            %9 = arith.maximumf %8, %arg6 : f32
            affine.yield %9 : f32
          }
          affine.yield %7 : f32
        }
        affine.store %3, %wram_buf_1[symbol(%0)] : memref<16xf32, "wram">
        threads.barrier
        threads.on_thread 0 {
          %6 = affine.load %wram_buf_2[] : memref<f32, "wram">
          %7 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %6) -> (f32) {
            %8 = affine.load %wram_buf_1[%arg3] : memref<16xf32, "wram">
            %9 = arith.maximumf %8, %arg4 : f32
            affine.yield %9 : f32
          }
          affine.store %7, %wram_buf_2[] : memref<f32, "wram">
        }
        threads.on_thread 1 {
          affine.store %cst, %wram_buf[] : memref<f32, "wram">
        }
        threads.on_thread 0 {
          affine.for %arg3 = 0 to 16 {
            affine.store %cst, %wram_buf_1[%arg3] : memref<16xf32, "wram">
          }
        }
        threads.barrier
        %4 = affine.load %wram_buf_2[] : memref<f32, "wram">
        affine.for %arg3 = 0 to 8 {
          threads.barrier
          %6 = affine.apply #map(%arg3)[%0]
          %subview_4 = memref.subview %arg2[%6] [512] [1] : memref<65536xf32, "mram"> to memref<512xf32, strided<[1], offset: ?>, "mram">
          upmem.local_transfer %subview_4 into %subview : memref<512xf32, strided<[1], offset: ?>, "mram"> to memref<512xf32, strided<[1], offset: ?>, "wram">
          %7 = affine.load %wram_buf_1[symbol(%0)] : memref<16xf32, "wram">
          %8 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %7) -> (f32) {
            %9 = affine.load %wram_buf_0[%arg4 + symbol(%0) * 512] : memref<8192xf32, "wram">
            %10 = arith.subf %9, %4 : f32
            %11 = math.exp %10 : f32
            affine.store %11, %wram_buf_0[%arg4 + symbol(%0) * 512] : memref<8192xf32, "wram">
            %12 = arith.addf %11, %arg5 : f32
            affine.yield %12 : f32
          }
          affine.store %8, %wram_buf_1[symbol(%0)] : memref<16xf32, "wram">
          upmem.local_transfer %subview into %subview_4 : memref<512xf32, strided<[1], offset: ?>, "wram"> to memref<512xf32, strided<[1], offset: ?>, "mram">
        }
        threads.barrier
        threads.on_thread 0 {
          %6 = affine.load %wram_buf[] : memref<f32, "wram">
          %7 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %6) -> (f32) {
            %8 = affine.load %wram_buf_1[%arg3] : memref<16xf32, "wram">
            %9 = arith.addf %8, %arg4 : f32
            affine.yield %9 : f32
          }
          affine.store %7, %wram_buf[] : memref<f32, "wram">
        }
        threads.barrier
        %5 = affine.load %wram_buf[] : memref<f32, "wram">
        affine.for %arg3 = 0 to 8 {
          threads.barrier
          %6 = affine.apply #map(%arg3)[%0]
          %subview_4 = memref.subview %arg2[%6] [512] [1] : memref<65536xf32, "mram"> to memref<512xf32, strided<[1], offset: ?>, "mram">
          upmem.local_transfer %subview_4 into %subview : memref<512xf32, strided<[1], offset: ?>, "mram"> to memref<512xf32, strided<[1], offset: ?>, "wram">
          affine.for %arg4 = 0 to 512 {
            %7 = affine.load %wram_buf_0[%arg4 + symbol(%0) * 512] : memref<8192xf32, "wram">
            %8 = arith.divf %7, %5 : f32
            affine.store %8, %wram_buf_0[%arg4 + symbol(%0) * 512] : memref<8192xf32, "wram">
          }
          upmem.local_transfer %subview into %subview_4 : memref<512xf32, strided<[1], offset: ?>, "wram"> to memref<512xf32, strided<[1], offset: ?>, "mram">
        }
      }
    }
    transfer %buf into %arg0 : <65536xf32, mram(r, d)> to host
    transform.sequence  failures(propagate) {
    ^bb0(%arg1: !transform.op<"btfl.block">):
      transform.btfl.lower.finish_accelerator_lowering %arg1
    }
  }
}

