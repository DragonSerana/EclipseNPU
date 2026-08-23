// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.act' op ActOp src and dst shapes must match

module {
  func.func @bad(%src: memref<4x4xf16, 0>, %dst: memref<4x8xf16, 0>) {
    eclipse.act %src, %dst {kind = relu}
      : memref<4x4xf16, 0>, memref<4x8xf16, 0>
    return
  }
}
