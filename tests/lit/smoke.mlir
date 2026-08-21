// RUN: %mlir-opt --canonicalize %s | %FileCheck %s

func.func @smoke(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = arith.addi %x, %c0 : i32
  return %r : i32
}

// CHECK-LABEL: func.func @smoke
// CHECK: return
