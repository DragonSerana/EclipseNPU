// RUN: %not %eclipse-opt %s > %t 2>&1
// RUN: %FileCheck %s < %t
// CHECK: expected string or keyword containing one of the following enum values for attribute 'kind' [relu]

module {
  func.func @bad(%src: memref<4x4xf16, 0>, %dst: memref<4x4xf16, 0>) {
    eclipse.act %src, %dst {kind = gelu}
      : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
