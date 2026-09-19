// RUN: %eclipse-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-eclipse | %FileCheck %s

// rowwise 归约 [rows, cols] -> [rows, 1]，kind 从 combiner body 认出来。
// CHECK: eclipse.reduce {{.*}}kind = sum
// CHECK: eclipse.reduce {{.*}}kind = square_sum
// CHECK: eclipse.reduce {{.*}}kind = max

func.func @reduce_sum(%A: tensor<4x8xf16>) -> tensor<4x1xf16> {
  %init = tensor.empty() : tensor<4x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<4x8xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %s = arith.addf %in, %out : f16
    linalg.yield %s : f16
  } -> tensor<4x1xf16>
  return %r : tensor<4x1xf16>
}

func.func @reduce_square_sum(%A: tensor<4x8xf16>) -> tensor<4x1xf16> {
  %init = tensor.empty() : tensor<4x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<4x8xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %sq = arith.mulf %in, %in : f16
    %s = arith.addf %out, %sq : f16
    linalg.yield %s : f16
  } -> tensor<4x1xf16>
  return %r : tensor<4x1xf16>
}

func.func @reduce_max(%A: tensor<4x8xf16>) -> tensor<4x1xf16> {
  %init = tensor.empty() : tensor<4x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<4x8xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %m = arith.maximumf %in, %out : f16
    linalg.yield %m : f16
  } -> tensor<4x1xf16>
  return %r : tensor<4x1xf16>
}
