// ACT EXP：linalg.exp 具名 op -> eclipse.act kind=exp。
func.func @act_exp_16(%A: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %init = tensor.empty() : tensor<16x16xf16>
  %R = linalg.exp ins(%A : tensor<16x16xf16>)
                   outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %R : tensor<16x16xf16>
}
