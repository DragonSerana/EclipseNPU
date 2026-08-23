// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.elementwise_add' op EwiseAddOp lhs and rhs shapes must match

module {
  func.func @bad(%a: memref<4x4xf16, 0>, %b: memref<4x8xf16, 0>,
                 %c: memref<4x4xf16, 0>) {
    eclipse.elementwise_add %a, %b, %c
      : memref<4x4xf16, 0>, memref<4x8xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
