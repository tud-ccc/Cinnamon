#upmem = #upmem.platform<type = v1A, dimensions = 40x64x24>



func.func @gemv(%A: tensor<768x768xf32>, %x: tensor<768xf32>) -> tensor<768xf32> {
  %4 = cinm.compute -> tensor<768xf32> attributes {cinm.available_platforms = [#upmem]} {
        %18 = cinm.op.gemv %A, %x : tensor<768x768xf32>, tensor<768xf32> -> tensor<768xf32>
        cinm.yield %18 : tensor<768xf32>
  }
  return %4 : tensor<768xf32>
}