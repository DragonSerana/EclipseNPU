// RUN: %not %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: failed to legalize operation 'linalg.generic'

// body 是 x*x，不是任何已知激活。ActLowering 认不出来 → 这个 linalg op 降不掉，
// 而 pass 把 linalg dialect 标成了 illegal，所以 conversion framework 直接报错。
// 关键是不能静默放过：漏掉的算子会让 .easm 少一条指令、结果错却不报错。
func.func @unknown_act(%A: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %i0 = tensor.empty() : tensor<4x8xf16>
  %0 = linalg.generic {indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%A : tensor<4x8xf16>) outs(%i0 : tensor<4x8xf16>) {
    ^bb0(%in: f16, %out: f16):
      %sq = arith.mulf %in, %in : f16
      linalg.yield %sq : f16
  } -> tensor<4x8xf16>
  return %0 : tensor<4x8xf16>
}
