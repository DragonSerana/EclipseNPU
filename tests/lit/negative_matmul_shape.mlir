// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: error: 'eclipse.matmul' op MatmulOp lhs K must match rhs M

module {
  func.func @bad(%a: memref<4x8xf16, 0>, %b: memref<4x4xf16, 0>,
                 %c: memref<4x4xf16, 0>) {
    eclipse.matmul %a, %b, %c {accumulate = true}
      : memref<4x8xf16, 0>, memref<4x4xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
