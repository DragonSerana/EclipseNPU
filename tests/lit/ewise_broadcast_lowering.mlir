// RUN: %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse | %FileCheck %s

// 三种广播形态都降到 eclipse.elementwise_mul，rhs 保持缩小后的 shape（读模式
// 由 shape 推出，见 docs/spec/isa.md 的 EwiseParam）。
// CHECK: eclipse.elementwise_mul {{.*}} : memref<4x64xf16>, memref<1x64xf16>, memref<4x64xf16>
// CHECK: eclipse.elementwise_mul {{.*}} : memref<4x64xf16>, memref<4x1xf16>, memref<4x64xf16>
// CHECK: eclipse.elementwise_mul {{.*}} : memref<4x64xf16>, memref<4x32xf16>, memref<4x64xf16>

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (0, d1)>
#col = affine_map<(d0, d1) -> (d0, 0)>
#blk = affine_map<(d0, d1) -> (d0, d1 mod 32)>

func.func @bcast(%x: tensor<4x64xf16>, %gamma: tensor<1x64xf16>,
                 %inv: tensor<4x1xf16>, %cos: tensor<4x32xf16>) -> tensor<4x64xf16> {
  %i0 = tensor.empty() : tensor<4x64xf16>
  %0 = linalg.generic {indexing_maps = [#id, #row, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%x, %gamma : tensor<4x64xf16>, tensor<1x64xf16>)
      outs(%i0 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %m = arith.mulf %a, %b : f16
      linalg.yield %m : f16
  } -> tensor<4x64xf16>

  %i1 = tensor.empty() : tensor<4x64xf16>
  %1 = linalg.generic {indexing_maps = [#id, #col, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%0, %inv : tensor<4x64xf16>, tensor<4x1xf16>)
      outs(%i1 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %m = arith.mulf %a, %b : f16
      linalg.yield %m : f16
  } -> tensor<4x64xf16>

  %i2 = tensor.empty() : tensor<4x64xf16>
  %2 = linalg.generic {indexing_maps = [#id, #blk, #id],
                       iterator_types = ["parallel", "parallel"]}
      ins(%1, %cos : tensor<4x64xf16>, tensor<4x32xf16>)
      outs(%i2 : tensor<4x64xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %m = arith.mulf %a, %b : f16
      linalg.yield %m : f16
  } -> tensor<4x64xf16>

  return %2 : tensor<4x64xf16>
}
