// REDUCE MAX: A[128,128] 沿 K 取最大值成 [128,1]（softmax 的稳定化减数）。
func.func @reduce_max_128(%A: tensor<128x128xf16>) -> tensor<128x1xf16> {
  %init = tensor.empty() : tensor<128x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<128x128xf16>) outs(%init : tensor<128x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %m = arith.maximumf %in, %out : f16
    linalg.yield %m : f16
  } -> tensor<128x1xf16>
  return %r : tensor<128x1xf16>
}
