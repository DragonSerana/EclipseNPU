// ACT RSQRT：linalg.rsqrt 具名 op -> eclipse.act kind=rsqrt。
func.func @act_rsqrt_128(%A: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init = tensor.empty() : tensor<128x128xf16>
  %R = linalg.rsqrt ins(%A : tensor<128x128xf16>)
                    outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %R : tensor<128x128xf16>
}
