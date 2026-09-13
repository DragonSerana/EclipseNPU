func.func @ewise_add_16(%A: tensor<16x16xf16>, %B: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %init = tensor.empty() : tensor<16x16xf16>
  %C = linalg.add ins(%A, %B : tensor<16x16xf16>, tensor<16x16xf16>)
                   outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %C : tensor<16x16xf16>
}
