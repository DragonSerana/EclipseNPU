// RUN: %not %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse > %t 2>&1
// RUN: %FileCheck %s < %t
// rhs 的行数既不是 1 也不是 dst 的行数，不是受支持的广播形态，不该降下去。
// CHECK: failed to legalize operation 'linalg.generic'

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (0, d1)>

func.func @bad_bcast(%x: tensor<4x64xf16>, %b: tensor<2x64xf16>) -> tensor<4x64xf16> {
  %i0 = tensor.empty() : tensor<4x64xf16>
  %0 = linalg.generic {indexing_maps = [#id, #row, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %b : tensor<4x64xf16>, tensor<2x64xf16>)
      outs(%i0 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %c: f16, %d: f16):
      %m = arith.mulf %a, %c : f16
      linalg.yield %m : f16
  } -> tensor<4x64xf16>
  return %0 : tensor<4x64xf16>
}
