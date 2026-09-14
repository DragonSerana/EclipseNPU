// RUN: %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse | %FileCheck %s

// linalg.exp / linalg.rsqrt 是具名 op，要降到带对应 kind 的 eclipse.act。
// CHECK: eclipse.act {{.*}}kind = exp
// CHECK: eclipse.act {{.*}}kind = rsqrt

func.func @act_lowering(%A: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %i0 = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.exp ins(%A : tensor<4x8xf16>) outs(%i0 : tensor<4x8xf16>) -> tensor<4x8xf16>
  %i1 = tensor.empty() : tensor<4x8xf16>
  %1 = linalg.rsqrt ins(%0 : tensor<4x8xf16>) outs(%i1 : tensor<4x8xf16>) -> tensor<4x8xf16>
  return %1 : tensor<4x8xf16>
}
