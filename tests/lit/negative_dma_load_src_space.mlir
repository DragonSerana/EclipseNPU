// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.dma_load' op operand #1 must be 2D f16 memref in DDR

module {
  func.func @bad(%dst: memref<4x4xf16, 0>, %src: memref<4x4xf16, 0>) {
    eclipse.dma_load %dst, %src : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
