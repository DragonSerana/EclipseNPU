// RUN: %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse | %FileCheck %s

// 具名 op linalg.reduce：输出是 rank-1 的 [rows]，降级时要摊成 [rows,1]。
// CHECK: eclipse.reduce {{.*}}kind = sum
// CHECK: eclipse.reduce {{.*}}kind = max

func.func @named_sum(%A: tensor<4x8xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.reduce ins(%A : tensor<4x8xf16>) outs(%init : tensor<4xf16>)
      dimensions = [1]
      (%in: f16, %out: f16) {
    %s = arith.addf %in, %out : f16
    linalg.yield %s : f16
  }
  return %r : tensor<4xf16>
}

func.func @named_max(%A: tensor<4x8xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.reduce ins(%A : tensor<4x8xf16>) outs(%init : tensor<4xf16>)
      dimensions = [1]
      (%in: f16, %out: f16) {
    %m = arith.maximumf %in, %out : f16
    linalg.yield %m : f16
  }
  return %r : tensor<4xf16>
}
