// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.dma_load' op dst (SRAM) must have identity layout (no strides)

module {
  func.func @bad(%dst: memref<4x4xf16, strided<[8, 1]>, 0>,
                 %src: memref<4x4xf16, 1>) {
    eclipse.dma_load %dst, %src : memref<4x4xf16, strided<[8, 1]>, 0>, memref<4x4xf16, 1>
    return
  }
}
