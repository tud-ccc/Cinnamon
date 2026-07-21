  func.func @oo() -> tensor<128x4xi32> {
      %cst = arith.constant dense<0> : tensor<4x128xi32>

      %o = tensor.empty() : tensor<128x4xi32>
      %cst2 = linalg.transpose ins(%cst : tensor<4x128xi32>) outs(%o : tensor<128x4xi32>) permutation = [1, 0]
      return %cst2: tensor<128x4xi32>
  }