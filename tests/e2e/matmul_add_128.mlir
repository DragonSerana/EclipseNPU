func.func @matmul_add(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>, %bias: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init0 = tensor.empty() : tensor<128x128xf16>
  %C = linalg.matmul ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>)
                     outs(%init0 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %init1 = tensor.empty() : tensor<128x128xf16>
  %D = linalg.add ins(%C, %bias : tensor<128x128xf16>, tensor<128x128xf16>)
                   outs(%init1 : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %D : tensor<128x128xf16>
}
