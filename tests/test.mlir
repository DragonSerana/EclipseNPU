func.func @matmul(%A: tensor<16x16xf16>, %B: tensor<16x16xf16>) -> tensor<16x16xf16> {
  // 创建一个未初始化的输出 tensor（稍后由 matmul 填充）
  %init = tensor.empty() : tensor<16x16xf16>
  // linalg.matmul 使用 ins/outs，并显式标注类型
  %C = linalg.matmul ins(%A, %B : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
  return %C : tensor<16x16xf16>
}