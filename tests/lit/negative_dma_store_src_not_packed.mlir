// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.dma_store' op src (SRAM) must be packed

module {
  func.func @bad(%src: memref<4x4xf16, strided<[8, 1]>, 0>,
                 %dst: memref<4x4xf16, 1>) {
    eclipse.dma_store %src, %dst : memref<4x4xf16, strided<[8, 1]>, 0>, memref<4x4xf16, 1>
    return
  }
}
