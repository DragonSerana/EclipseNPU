func.func @matmul_128(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init0 = tensor.empty() : tensor<128x128xf16>
  %C = linalg.matmul ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>)
                     outs(%init0 : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %C : tensor<128x128xf16>
}
