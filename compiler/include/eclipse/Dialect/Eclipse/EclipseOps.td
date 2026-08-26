#ifndef ECLIPSE_OPS
#define ECLIPSE_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Eclipse_Dialect : Dialect {
  let name = "eclipse";
  let summary = "Eclipse NPU Dialect";
  let cppNamespace = "::mlir::eclipse";
}

class Eclipse_Op<string mnemonic, list<Trait> traits = []>
    : Op<Eclipse_Dialect, mnemonic, traits>;

// 与 EclipseConstants.h 保持同步：0 = SRAM，1 = DDR。
def Eclipse_SRAMMemRef : ConfinedType<
    MemRefRankOf<[F16], [2]>,
    [CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 0">],
    "2D f16 memref in SRAM", "::mlir::MemRefType">;

def Eclipse_DDRMemRef : ConfinedType<
    MemRefRankOf<[F16], [2]>,
    [CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 1">],
    "2D f16 memref in DDR", "::mlir::MemRefType">;

def Eclipse_ActKind_RELU : I32EnumAttrCase<"RELU", 0, "relu">;
def Eclipse_ActKindAttr : I32EnumAttr<"ActKind", "Eclipse activation kind",
    [Eclipse_ActKind_RELU]> {
  let cppNamespace = "::mlir::eclipse";
}

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
    Eclipse_DDRMemRef:$src,
    Eclipse_SRAMMemRef:$dst
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

  let assemblyFormat = "$dst `,` $src attr-dict `:` type($dst) `,` type($src)";
}

def Eclipse_MatmulOp : Eclipse_Op<"matmul"> {
  let summary = "Eclipse NPU matrix multiply";
  let description = [{
    `eclipse.matmul` corresponds to the MATMUL instruction in the ISA.
    It computes `dst = lhs * rhs` (or accumulates into `dst` when
    `accumulate` is true).  All operands are SRAM memrefs; the shapes must be
    lhs(MxK), rhs(KxN), dst(MxN).
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$lhs,
    Eclipse_SRAMMemRef:$rhs,
    Eclipse_SRAMMemRef:$dst,
    BoolAttr:$accumulate
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $dst attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst)";
}

def Eclipse_EwiseAddOp : Eclipse_Op<"elementwise_add"> {
  let summary = "Eclipse NPU elementwise add";
  let description = [{
    `eclipse.elementwise_add` corresponds to the ELEMENTWISE_ADD instruction
    in the ISA.  It adds `n` fp16 elements: `dst = lhs + rhs`.  All operands
    are SRAM memrefs with the same shape.
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$lhs,
    Eclipse_SRAMMemRef:$rhs,
    Eclipse_SRAMMemRef:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$lhs `,` $rhs `,` $dst attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst)";
}

def Eclipse_ActOp : Eclipse_Op<"act"> {
  let summary = "Eclipse NPU activation";
  let description = [{
    `eclipse.act` corresponds to the ACT instruction in the ISA.
    It applies the activation selected by `kind` to `n` fp16 elements:
    `dst = act(src)`.  Both operands are SRAM memrefs with the same shape.
    v0.1 only supports RELU (kind = 0).
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$src,
    Eclipse_SRAMMemRef:$dst,
    Eclipse_ActKindAttr:$kind
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$src `,` $dst `{` `kind` `=` $kind `}` attr-dict `:` type($src) `,` type($dst)";
}

def Eclipse_SyncOp : Eclipse_Op<"sync"> {
  let summary = "Eclipse NPU fence";
  let description = [{
    `eclipse.sync` corresponds to the SYNC instruction in the ISA.  It is a
    fence that waits for all previously issued instructions to finish.
  }];

  let arguments = (ins);
  let results = (outs);

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Eclipse_SramOp : Eclipse_Op<"sram"> {
  let summary = "Declare an SRAM buffer at a fixed address";
  let description = [{
    `eclipse.sram` represents a fixed-address SRAM buffer produced by
    `eclipse-allocate`. It returns a memref in memory_space 0.
  }];

  let arguments = (ins I32Attr:$addr);
  let results = (outs Eclipse_SRAMMemRef:$buffer);

  let assemblyFormat = "custom<PrintHex>($addr) attr-dict `:` type($buffer)";
}


#endif // ECLIPSE_OPS

