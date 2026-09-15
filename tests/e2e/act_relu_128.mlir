// ACT RELU：linalg 侧没有 relu 具名 op，前端会把它写成 maximumf(x, 0) 的 generic。
func.func @act_relu_128(%A: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init = tensor.empty() : tensor<128x128xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>,
                                        affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%A : tensor<128x128xf16>)
      outs(%init : tensor<128x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %zero = arith.constant 0.0 : f16
      %max = arith.maximumf %in, %zero : f16
      linalg.yield %max : f16
  } -> tensor<128x128xf16>
  return %R : tensor<128x128xf16>
}
