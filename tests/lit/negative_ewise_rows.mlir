// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.elementwise_add' op rhs rows must be 1 or match dst rows

module {
  func.func @bad(%a: memref<4x4xf16, 0>, %b: memref<2x4xf16, 0>,
                 %c: memref<4x4xf16, 0>) {
    eclipse.elementwise_add %a, %b, %c
      : memref<4x4xf16, 0>, memref<2x4xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
