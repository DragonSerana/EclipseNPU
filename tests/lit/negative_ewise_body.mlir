// RUN: %not %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse > %t 2>&1
// RUN: %FileCheck %s < %t
// body 必须恰好用 {lhs, rhs} 两个输入。用 out 自己、或者同一个输入用两次，
// 都会让 eclipse 指令的语义（dst = lhs op rhs）和 IR 对不上，必须拒绝而不是降下去。
// CHECK-COUNT-2: failed to legalize operation 'linalg.generic'

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (0, d1)>

func.func @out_plus_out(%x: tensor<4x64xf16>, %b: tensor<1x64xf16>) -> tensor<4x64xf16> {
  %i0 = tensor.empty() : tensor<4x64xf16>
  %0 = linalg.generic {indexing_maps = [#id, #row, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %b : tensor<4x64xf16>, tensor<1x64xf16>)
      outs(%i0 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %c: f16, %d: f16):
      %s = arith.addf %d, %d : f16
      linalg.yield %s : f16
  } -> tensor<4x64xf16>
  return %0 : tensor<4x64xf16>
}

func.func @rhs_plus_rhs(%x: tensor<4x64xf16>, %b: tensor<1x64xf16>) -> tensor<4x64xf16> {
  %i0 = tensor.empty() : tensor<4x64xf16>
  %0 = linalg.generic {indexing_maps = [#id, #row, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %b : tensor<4x64xf16>, tensor<1x64xf16>)
      outs(%i0 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %c: f16, %d: f16):
      %s = arith.addf %c, %c : f16
      linalg.yield %s : f16
  } -> tensor<4x64xf16>
  return %0 : tensor<4x64xf16>
}
