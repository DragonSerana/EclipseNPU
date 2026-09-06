module {
  func.func @matmul_128(%arg0: memref<128x128xf16, strided<[?, ?], offset: ?>> {eclipse.ddr_addr = 2147549184 : i64}, %arg1: memref<128x128xf16, strided<[?, ?], offset: ?>> {eclipse.ddr_addr = 2147614720 : i64}) -> memref<128x128xf16> {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 64 : i64, eclipse.ddr_addr = 2147680256 : i64} : memref<128x128xf16>
    %memspacecast = memref.memory_space_cast %arg0 : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<128x128xf16, strided<[?, ?], offset: ?>, 1>
    %memspacecast_0 = memref.memory_space_cast %arg1 : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<128x128xf16, strided<[?, ?], offset: ?>, 1>
    %memspacecast_1 = memref.memory_space_cast %alloc : memref<128x128xf16> to memref<128x128xf16, 1>
    %0 = eclipse.sram 0x10000000 : memref<128x16xf16>
    %1 = eclipse.sram 0x10001000 : memref<16x128xf16>
    %2 = eclipse.sram 0x10002000 : memref<128x128xf16>
    %subview = memref.subview %memspacecast[0, 0] [128, 16] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>, 1> to memref<128x16xf16, strided<[?, ?], offset: ?>, 1>
    %subview_2 = memref.subview %memspacecast_0[0, 0] [16, 128] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>, 1> to memref<16x128xf16, strided<[?, ?], offset: ?>, 1>
    eclipse.dma_load %0, %subview : memref<128x16xf16>, memref<128x16xf16, strided<[?, ?], offset: ?>, 1>
    eclipse.dma_load %1, %subview_2 : memref<16x128xf16>, memref<16x128xf16, strided<[?, ?], offset: ?>, 1>
    eclipse.sync
    eclipse.matmul %0, %1, %2 {accumulate = false} : memref<128x16xf16>, memref<16x128xf16>, memref<128x128xf16>
    eclipse.sync
    scf.for %arg2 = %c1 to %c8 step %c1 {
      %3 = arith.muli %arg2, %c16 : index
      %subview_3 = memref.subview %memspacecast[0, %3] [128, 16] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>, 1> to memref<128x16xf16, strided<[?, ?], offset: ?>, 1>
      %subview_4 = memref.subview %memspacecast_0[%3, 0] [16, 128] [1, 1] : memref<128x128xf16, strided<[?, ?], offset: ?>, 1> to memref<16x128xf16, strided<[?, ?], offset: ?>, 1>
      eclipse.dma_load %0, %subview_3 : memref<128x16xf16>, memref<128x16xf16, strided<[?, ?], offset: ?>, 1>
      eclipse.dma_load %1, %subview_4 : memref<16x128xf16>, memref<16x128xf16, strided<[?, ?], offset: ?>, 1>
      eclipse.sync
      eclipse.matmul %0, %1, %2 {accumulate = true} : memref<128x16xf16>, memref<16x128xf16>, memref<128x128xf16>
      eclipse.sync
    }
    eclipse.dma_store %memspacecast_1, %2 : memref<128x128xf16, 1>, memref<128x128xf16>
    memref.dealloc %2 : memref<128x128xf16>
    memref.dealloc %1 : memref<16x128xf16>
    memref.dealloc %0 : memref<128x16xf16>
    return %alloc : memref<128x128xf16>
  }
}

