// RUN: %eclipse-opt %s --one-shot-bufferize=bufferize-function-boundaries \
// RUN:   --convert-linalg-to-eclipse --eclipse-elide-copies --canonicalize \
// RUN:   --eclipse-allocate=layout=bump --eclipse-to-easm=output-easm=%t.easm
// RUN: cat %t.easm | %FileCheck %s

// 整条链路的编译侧回归：linalg.generic 归约 -> eclipse.reduce -> 分配 SRAM 地址
// -> 发 .easm。数值对拍在 tools/accuracy_check.py 的 reduce case（需要跑 cmodel）。
// CHECK: DMA_LOAD {{.*}}rows=8 cols=16
// CHECK: REDUCE {{.*}}rows=8 cols=16 kind=1
// CHECK: DMA_STORE {{.*}}rows=8 cols=1

func.func @reduce_sum_8x16(%A: tensor<8x16xf16>) -> tensor<8x1xf16> {
  %init = tensor.empty() : tensor<8x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<8x16xf16>) outs(%init : tensor<8x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %s = arith.addf %in, %out : f16
    linalg.yield %s : f16
  } -> tensor<8x1xf16>
  return %r : tensor<8x1xf16>
}

// CHECK: REDUCE {{.*}}rows=8 cols=16 kind=2
func.func @reduce_square_sum_8x16(%A: tensor<8x16xf16>) -> tensor<8x1xf16> {
  %init = tensor.empty() : tensor<8x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<8x16xf16>) outs(%init : tensor<8x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %sq = arith.mulf %in, %in : f16
    %s = arith.addf %out, %sq : f16
    linalg.yield %s : f16
  } -> tensor<8x1xf16>
  return %r : tensor<8x1xf16>
}

// CHECK: REDUCE {{.*}}rows=8 cols=16 kind=0
func.func @reduce_max_8x16(%A: tensor<8x16xf16>) -> tensor<8x1xf16> {
  %init = tensor.empty() : tensor<8x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, 0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%A : tensor<8x16xf16>) outs(%init : tensor<8x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %m = arith.maximumf %in, %out : f16
    linalg.yield %m : f16
  } -> tensor<8x1xf16>
  return %r : tensor<8x1xf16>
}
