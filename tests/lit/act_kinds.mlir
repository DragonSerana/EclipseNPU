// RUN: %eclipse-opt %s | %FileCheck %s

// ACT 的 kind 枚举：relu / exp / rsqrt / silu 都要能解析并原样打印。
// CHECK: eclipse.act {{.*}}kind = relu
// CHECK: eclipse.act {{.*}}kind = exp
// CHECK: eclipse.act {{.*}}kind = rsqrt
// CHECK: eclipse.act {{.*}}kind = silu

module {
  func.func @act_kinds(%src: memref<4x4xf16, 0>, %dst: memref<4x4xf16, 0>) {
    eclipse.act %src, %dst {kind = relu} : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    eclipse.act %src, %dst {kind = exp} : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    eclipse.act %src, %dst {kind = rsqrt} : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    eclipse.act %src, %dst {kind = silu} : memref<4x4xf16, 0>, memref<4x4xf16, 0>
    return
  }
}
