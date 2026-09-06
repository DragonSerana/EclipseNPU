func.func @matmul_add_relu(%A: tensor<16x16xf16>, %B: tensor<16x16xf16>, %bias: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %init0 = tensor.empty() : tensor<16x16xf16>
  %C = linalg.matmul ins(%A, %B : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%init0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %init1 = tensor.empty() : tensor<16x16xf16>
  %D = linalg.add ins(%C, %bias : tensor<16x16xf16>, tensor<16x16xf16>)
                   outs(%init1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %init2 = tensor.empty() : tensor<16x16xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%D : tensor<16x16xf16>)
      outs(%init2 : tensor<16x16xf16>) {
    ^bb0(%in: f16, %out: f16):
      %zero = arith.constant 0.0 : f16
      %max = arith.maximumf %in, %zero : f16
      linalg.yield %max : f16
  } -> tensor<16x16xf16>
  return %R : tensor<16x16xf16>
}
