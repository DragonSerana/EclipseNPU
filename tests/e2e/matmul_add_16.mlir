func.func @matmul_add(%A: tensor<16x16xf16>, %B: tensor<16x16xf16>, %bias: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %init0 = tensor.empty() : tensor<16x16xf16>
  %C = linalg.matmul ins(%A, %B : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%init0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %init1 = tensor.empty() : tensor<16x16xf16>
  %D = linalg.add ins(%C, %bias : tensor<16x16xf16>, tensor<16x16xf16>)
                   outs(%init1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %D : tensor<16x16xf16>
}
