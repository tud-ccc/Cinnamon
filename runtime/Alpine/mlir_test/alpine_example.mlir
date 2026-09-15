module {
  // MLIR counterpart of runtime/Alpine/example.cc.
  // Takes float domain tensors, handles quantization/dequantization, and
  // streams data through the Alpine runtime primitives.
  func.func @alpine_example(%input_f: memref<32xf32>,
                            %weights_f: memref<256x256xf32>,
                            %output_f: memref<32xf32>) {
    %input_q = memref.alloc() : memref<32xi8>
    %weights_q = memref.alloc() : memref<256x256xi8>
    %output_q = memref.alloc() : memref<32xi8>

    alpine.quantize %input_f, %input_q {scale = 1.0 : f32, zero = 0 : i32}
        : memref<32xf32>, memref<32xi8>
    alpine.quantize %weights_f, %weights_q {scale = 1.0 : f32, zero = 0 : i32}
        : memref<256x256xf32>, memref<256x256xi8>

    %tile = alpine.alloc_tile {height = 256, width = 256} -> i32
    alpine.write_weights %tile, %weights_q : i32, memref<256x256xi8>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_step = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1_step {
      alpine.enqueue_vec %tile, %input_q : i32, memref<32xi8>
      alpine.process %tile { count = 1 } : i32
      alpine.dequeue_vec %tile, %output_q : i32, memref<32xi8>
      scf.yield
    }

    alpine.dequantize %output_q, %output_f {scale = 1.0 : f32, zero = 0 : i32}
        : memref<32xi8>, memref<32xf32>

    memref.dealloc %input_q : memref<32xi8>
    memref.dealloc %weights_q : memref<256x256xi8>
    memref.dealloc %output_q : memref<32xi8>
    func.return
  }
}
