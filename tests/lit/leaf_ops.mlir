// RUN: %eclipse-opt %s | %FileCheck %s

// CHECK-LABEL: func.func @all_ops
// CHECK: eclipse.dma_load
// CHECK: eclipse.dma_store
// CHECK: eclipse.matmul
// CHECK: eclipse.elementwise_add
// CHECK: eclipse.act
// CHECK: eclipse.sync
module {
  func.func @all_ops(
      %a: memref<128x16xf16, 0>,
      %b: memref<16x128xf16, 0>,
      %c: memref<128x128xf16, 0>,
      %tmp: memref<128x128xf16, 0>,
      %a_ddr: memref<128x16xf16, strided<[128, 1]>, 1>,
      %b_ddr: memref<16x128xf16, strided<[128, 1]>, 1>,
      %c_ddr: memref<128x128xf16, 1>) {
    eclipse.dma_load %a, %a_ddr : memref<128x16xf16, 0>, memref<128x16xf16, strided<[128, 1]>, 1>
    eclipse.dma_load %b, %b_ddr : memref<16x128xf16, 0>, memref<16x128xf16, strided<[128, 1]>, 1>
    eclipse.dma_store %c_ddr, %c : memref<128x128xf16, 1>, memref<128x128xf16, 0>
    eclipse.matmul %a, %b, %c {accumulate = true} : memref<128x16xf16, 0>, memref<16x128xf16, 0>, memref<128x128xf16, 0>
    eclipse.elementwise_add %c, %tmp, %tmp : memref<128x128xf16, 0>, memref<128x128xf16, 0>, memref<128x128xf16, 0>
    eclipse.act %tmp, %c {kind = relu} : memref<128x128xf16, 0>, memref<128x128xf16, 0>
    eclipse.sync
    return
  }
}