// RUN: %eclipse-opt %s --eclipse-to-easm=output-easm=%t.easm
// RUN: cat %t.easm | %FileCheck %s

// MATMUL 的转置标志：转置时 lhs 按 [K,M]、rhs 按 [N,K] 摆布，逻辑上仍算 MxN。
// 所以发出来的 M/K/N 要按逻辑形状算：lhs[8,4] + transA => M=4 K=8。
// CHECK: MATMUL {{.*}}M=4 N=16 K=8 acc=0 ta=1 tb=0
// CHECK: MATMUL {{.*}}M=4 N=16 K=8 acc=0 ta=0 tb=1
// CHECK: MATMUL {{.*}}M=4 N=16 K=8 acc=0 ta=1 tb=1

func.func @trans_a() {
  %a = eclipse.sram 0x10000000 : memref<8x4xf16>
  %b = eclipse.sram 0x10001000 : memref<8x16xf16>
  %c = eclipse.sram 0x10002000 : memref<4x16xf16>
  eclipse.matmul %a, %b, %c {accumulate = false, transA = true} : memref<8x4xf16>, memref<8x16xf16>, memref<4x16xf16>
  return
}

func.func @trans_b() {
  %a = eclipse.sram 0x10000000 : memref<4x8xf16>
  %b = eclipse.sram 0x10001000 : memref<16x8xf16>
  %c = eclipse.sram 0x10002000 : memref<4x16xf16>
  eclipse.matmul %a, %b, %c {accumulate = false, transB = true} : memref<4x8xf16>, memref<16x8xf16>, memref<4x16xf16>
  return
}

func.func @trans_ab() {
  %a = eclipse.sram 0x10000000 : memref<8x4xf16>
  %b = eclipse.sram 0x10001000 : memref<16x8xf16>
  %c = eclipse.sram 0x10002000 : memref<4x16xf16>
  eclipse.matmul %a, %b, %c {accumulate = false, transA = true, transB = true} : memref<8x4xf16>, memref<16x8xf16>, memref<4x16xf16>
  return
}
