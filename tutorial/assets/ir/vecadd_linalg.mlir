module {
  func.func @vecadd(%a: tensor<104857600xf32>, %b: tensor<104857600xf32>) -> tensor<104857600xf32> {
    %init = tensor.empty() : tensor<104857600xf32>

    %out = linalg.add
      ins(%a, %b : tensor<104857600xf32>, tensor<104857600xf32>)
      outs(%init : tensor<104857600xf32>) -> tensor<104857600xf32>

    return %out : tensor<104857600xf32>
  }
}
