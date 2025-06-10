
func.func @conv(%img : tensor<1x128x128x3xi16>, %flt : tensor<3x3x3x8xi16>, %bias:  tensor<1x126x126x8xi16>) {
  %conv = linalg.conv_2d_nhwc_hwcf
    ins(%img, %flt: tensor<1x128x128x3xi16>, tensor<3x3x3x8xi16>)
    outs(%bias: tensor<1x126x126x8xi16>)
    -> tensor<1x126x126x8xi16>
  return
}
