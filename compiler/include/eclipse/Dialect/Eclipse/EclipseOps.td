#ifndef ECLIPSE_OPS
#define ECLIPSE_OPS

include "mlir/IR/OpBase.td"

def Eclipse_Dialect : Dialect {
  let name = "eclipse";
  let summary = "Eclipse NPU Dialect";
  let cppNamespace = "::mlir::eclipse";
}

class Eclipse_Op<string mnemonic, list<Trait> traits = []>
    : Op<Eclipse_Dialect, mnemonic, traits>;

def Eclipse_SRAMMemRef : ConfinedType<
    MemRefRankOf<[F16], [2]>,
    [CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 0">],
    "2D f16 memref in SRAM", "::mlir::MemRefType">;

def Eclipse_DDRMemRef : ConfinedType<
    MemRefRankOf<[F16], [2]>,
    [CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 1">],
    "2D f16 memref in DDR", "::mlir::MemRefType">;


// 六个 leaf op 的定义将在这里逐个补充：
//   eclipse.dma_load
//   eclipse.dma_store
//   eclipse.matmul
//   eclipse.elementwise_add
//   eclipse.act
//   eclipse.sync

def Eclipse_DmaLoadOp : Eclipse_Op<"dma_load"> {
  let summary = "从 DDR 搬运一个二维 tile 到 SRAM";
  let description = [{
    `eclipse.dma_load` 对应 ISA 的 DMA_LOAD。它把 DDR 中的一个二维子视图
    搬运到 SRAM 的 packed 缓冲区。`dst` 必须是 memory_space 0（SRAM），
    `src` 必须是 memory_space 1（DDR）。rows/cols/stride 由 memref 形状和
    stride 推导，具体地址在后续 allocate/easm 阶段确定。
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$dst,
    Eclipse_DDRMemRef:$src
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src)";
}

#endif // ECLIPSE_OPS

