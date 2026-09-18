//#! cinm-opt --cinm-assign-platforms --cinm-isolate-compute-blocks --upmem-infer-accelerator --split-input-file
//
// (z, t) = (W2 * (W1 * x), W3 * x): a two-gemv chain plus one gemv parallel
// to it, sharing the activation. Size class = bytes of each i32 weight (see
// 2mv_seq.mlir); all three weights are static. W1 and W3 have the same
// shape and W2 the transposed one, so the graph holds two program classes,
// one of them with two members.
// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_1MB(%x: tensor<512xi32>, %W1: tensor<512x512xi32>{cinm.static}, %W2: tensor<512x512xi32>{cinm.static}, %W3: tensor<512x512xi32>{cinm.static}) -> (tensor<512xi32>, tensor<512xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %z = cinm.op.gemv %W2, %y : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    %t = cinm.op.gemv %W3, %x : tensor<512x512xi32>, tensor<512xi32> -> tensor<512xi32>
    func.return %z, %t : tensor<512xi32>, tensor<512xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_64MB(%x: tensor<4096xi32>, %W1: tensor<4096x4096xi32>{cinm.static}, %W2: tensor<4096x4096xi32>{cinm.static}, %W3: tensor<4096x4096xi32>{cinm.static}) -> (tensor<4096xi32>, tensor<4096xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %z = cinm.op.gemv %W2, %y : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    %t = cinm.op.gemv %W3, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    func.return %z, %t : tensor<4096xi32>, tensor<4096xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_256MB(%x: tensor<8192xi32>, %W1: tensor<8192x8192xi32>{cinm.static}, %W2: tensor<8192x8192xi32>{cinm.static}, %W3: tensor<8192x8192xi32>{cinm.static}) -> (tensor<8192xi32>, tensor<8192xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %z = cinm.op.gemv %W2, %y : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    %t = cinm.op.gemv %W3, %x : tensor<8192x8192xi32>, tensor<8192xi32> -> tensor<8192xi32>
    func.return %z, %t : tensor<8192xi32>, tensor<8192xi32>
}

// -----
#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 24>

func.func @_3mv_512MB(%x: tensor<16384xi32>, %W1: tensor<8192x16384xi32>{cinm.static}, %W2: tensor<16384x8192xi32>{cinm.static}, %W3: tensor<8192x16384xi32>{cinm.static}) -> (tensor<16384xi32>, tensor<8192xi32>)
attributes { cinm.available_platforms = [#upmem] } {
    %y = cinm.op.gemv %W1, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    %z = cinm.op.gemv %W2, %y : tensor<16384x8192xi32>, tensor<8192xi32> -> tensor<16384xi32>
    %t = cinm.op.gemv %W3, %x : tensor<8192x16384xi32>, tensor<16384xi32> -> tensor<8192xi32>
    func.return %z, %t : tensor<16384xi32>, tensor<8192xi32>
}
