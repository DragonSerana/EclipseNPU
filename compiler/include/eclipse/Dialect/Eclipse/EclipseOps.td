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
  let summary = "Copy a 2D tile from DDR to SRAM";
  let description = [{
    `eclipse.dma_load` corresponds to the DMA_LOAD instruction in the ISA.
    It copies a 2D subview from DDR into a packed buffer in SRAM.
    `dst` must be in memory_space 0 (SRAM), and `src` must be in memory_space 1 (DDR).
    The rows/cols/stride are derived from the shape and stride of the memrefs.
    The actual addresses are determined at later allocate/easm phases.
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$dst,
    Eclipse_DDRMemRef:$src
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src)";
}

def Eclipse_DmaStoreOp : Eclipse_Op<"dma_store"> {
  let summary = "Copy a 2D tile from SRAM to DDR";
  let description = [{
    `eclipse.dma_store` corresponds to the DMA_STORE instruction in the ISA.
    It copies a packed tile from SRAM into a 2D subview in DDR.
    `src` must be in memory_space 0 (SRAM), and `dst` must be in memory_space 1 (DDR).
    The rows/cols/stride are derived from the shape and stride of the memrefs.
    The actual addresses are determined at later allocate/easm phases.
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$src,
    Eclipse_DDRMemRef:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}


#endif // ECLIPSE_OPS

