// RUN: %eclipse-opt %s --verify-diagnostics

// dst 必须是 [rows, 1]
func.func @bad_dst(%A: memref<4x8xf16>, %D: memref<4x8xf16>) {
  // expected-error @+1 {{ReduceOp dst must be rows x 1}}
  eclipse.reduce %A, %D {kind = sum} : memref<4x8xf16>, memref<4x8xf16>
  return
}

// argmax 必须带索引缓冲
func.func @argmax_without_idx(%A: memref<4x8xf16>, %D: memref<4x1xf16>) {
  // expected-error @+1 {{ReduceOp idx is required by argmax, unused otherwise}}
  eclipse.reduce %A, %D {kind = argmax} : memref<4x8xf16>, memref<4x1xf16>
  return
}

// 非 argmax 不许带索引缓冲
func.func @sum_with_idx(%A: memref<4x8xf16>, %D: memref<4x1xf16>, %I: memref<4x1xi32>) {
  // expected-error @+1 {{ReduceOp idx is required by argmax, unused otherwise}}
  eclipse.reduce %A, %D, %I {kind = sum} : memref<4x8xf16>, memref<4x1xf16>, memref<4x1xi32>
  return
}

// src 和 dst 不能是同一块
func.func @alias(%A: memref<4x1xf16>) {
  // expected-error @+1 {{ReduceOp src/dst must not alias}}
  eclipse.reduce %A, %A {kind = sum} : memref<4x1xf16>, memref<4x1xf16>
  return
}
