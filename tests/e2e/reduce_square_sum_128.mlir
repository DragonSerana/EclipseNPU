// REDUCE SQUARE_SUM: A[128,128] 沿 K 归约 sum(x^2) 成 [128,1]（RMSNorm 的 mean(x^2)）。
func.func @reduce_square_sum_128(%A: tensor<128x128xf16>) -> tensor<128x1xf16> {
  %init = tensor.empty() : tensor<128x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<128x128xf16>) outs(%init : tensor<128x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %sq = arith.mulf %in, %in : f16
    %s = arith.addf %out, %sq : f16
    linalg.yield %s : f16
  } -> tensor<128x1xf16>
  return %r : tensor<128x1xf16>
}
