func.func @matmul_128_relu(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>, %bias: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init0 = tensor.empty() : tensor<128x128xf16>
  %C = linalg.matmul ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>)
                     outs(%init0 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %init1 = tensor.empty() : tensor<128x128xf16>
  %D = linalg.add ins(%C, %bias : tensor<128x128xf16>, tensor<128x128xf16>)
                   outs(%init1 : tensor<128x128xf16>) -> tensor<128x128xf16>
  %init2 = tensor.empty() : tensor<128x128xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%D : tensor<128x128xf16>)
      outs(%init2 : tensor<128x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %zero = arith.constant 0.0 : f16
      %max = arith.maximumf %in, %zero : f16
      linalg.yield %max : f16
  } -> tensor<128x128xf16>
  return %R : tensor<128x128xf16>
}
