// ACT EXP：linalg.exp 具名 op -> eclipse.act kind=exp。
func.func @act_exp_128(%A: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init = tensor.empty() : tensor<128x128xf16>
  %R = linalg.exp ins(%A : tensor<128x128xf16>)
                   outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %R : tensor<128x128xf16>
}
