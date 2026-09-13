func.func @ewise_div_128(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %init = tensor.empty() : tensor<128x128xf16>
  %C = linalg.div ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>)
                   outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %C : tensor<128x128xf16>
}
