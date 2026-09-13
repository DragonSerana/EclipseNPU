// RUN: %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse | %FileCheck %s

// 四个二元逐元素算子都要降到对应的 eclipse op。
// CHECK: eclipse.elementwise_add
// CHECK: eclipse.elementwise_sub
// CHECK: eclipse.elementwise_mul
// CHECK: eclipse.elementwise_div

func.func @ewise(%A: tensor<4x8xf16>, %B: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %i0 = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.add ins(%A, %B : tensor<4x8xf16>, tensor<4x8xf16>)
                  outs(%i0 : tensor<4x8xf16>) -> tensor<4x8xf16>
  %i1 = tensor.empty() : tensor<4x8xf16>
  %1 = linalg.sub ins(%0, %B : tensor<4x8xf16>, tensor<4x8xf16>)
                  outs(%i1 : tensor<4x8xf16>) -> tensor<4x8xf16>
  %i2 = tensor.empty() : tensor<4x8xf16>
  %2 = linalg.mul ins(%1, %B : tensor<4x8xf16>, tensor<4x8xf16>)
                  outs(%i2 : tensor<4x8xf16>) -> tensor<4x8xf16>
  %i3 = tensor.empty() : tensor<4x8xf16>
  %3 = linalg.div ins(%2, %B : tensor<4x8xf16>, tensor<4x8xf16>)
                  outs(%i3 : tensor<4x8xf16>) -> tensor<4x8xf16>
  return %3 : tensor<4x8xf16>
}
