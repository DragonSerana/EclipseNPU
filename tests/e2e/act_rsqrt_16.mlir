// ACT RSQRT：linalg.rsqrt 具名 op -> eclipse.act kind=rsqrt。
func.func @act_rsqrt_16(%A: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %init = tensor.empty() : tensor<16x16xf16>
  %R = linalg.rsqrt ins(%A : tensor<16x16xf16>)
                    outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %R : tensor<16x16xf16>
}
