//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// (y1, y2) = (W1 * x, W2 * x): two independent gemvs sharing the activation
// -- llama's QKV projections. Size class = bytes of each M x K i32 weight
// (see 2mv_seq.mlir); both weights are static.
//
// The two weights have the same shape, so both gemvs are one program class:
// the allocator can put them in one group that owns the whole device and
// runs them one after the other, rather than splitting the device between
// two classes it may not merge. That is the difference this file holds
// against 2mv_seq.
// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_par_1MB(%x: tensor<512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}) -> (tensor<512xi32>, tensor<512xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y1 = cinm.op.gemv %W1, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %y2 = cinm.op.gemv %W2, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    func.return %y1, %y2 : tensor<512xi32>, tensor<512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_par_64MB(%x: tensor<4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}) -> (tensor<4096xi32>, tensor<4096xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y1 = cinm.op.gemv %W1, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %y2 = cinm.op.gemv %W2, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    func.return %y1, %y2 : tensor<4096xi32>, tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_par_256MB(%x: tensor<8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}) -> (tensor<8192xi32>, tensor<8192xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y1 = cinm.op.gemv %W1, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %y2 = cinm.op.gemv %W2, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    func.return %y1, %y2 : tensor<8192xi32>, tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_2mv_par_512MB(%x: tensor<16384xi32>, %W1: tensor<8192x16384xi32>{cinm.static}, %W2: tensor<8192x16384xi32>{cinm.static}) -> (tensor<8192xi32>, tensor<8192xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y1 = cinm.op.gemv %W1, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    %y2 = cinm.op.gemv %W2, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    func.return %y1, %y2 : tensor<8192xi32>, tensor<8192xi32>
}
