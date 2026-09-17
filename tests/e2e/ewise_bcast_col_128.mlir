// 广播：B 缩小，读模式由 shape 决定（见 docs/spec/isa.md 的 EwiseParam）。
#id = affine_map<(d0, d1) -> (d0, d1)>
#rhs = affine_map<(d0, d1) -> (d0, 0)>

func.func @ewise_bcast_col_128(%A: tensor<128x128xf16>, %B: tensor<128x1xf16>) -> tensor<128x128xf16> {
  %init = tensor.empty() : tensor<128x128xf16>
  %C = linalg.generic {indexing_maps = [#id, #rhs, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%A, %B : tensor<128x128xf16>, tensor<128x1xf16>)
      outs(%init : tensor<128x128xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %r = arith.mulf %a, %b : f16
      linalg.yield %r : f16
  } -> tensor<128x128xf16>
  return %C : tensor<128x128xf16>
}
