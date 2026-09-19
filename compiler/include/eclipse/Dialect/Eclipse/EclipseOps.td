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
def Eclipse_ActKind_EXP : I32EnumAttrCase<"EXP", 1, "exp">;
def Eclipse_ActKind_RSQRT : I32EnumAttrCase<"RSQRT", 2, "rsqrt">;
def Eclipse_ActKind_SILU : I32EnumAttrCase<"SILU", 3, "silu">;
def Eclipse_ActKindAttr : I32EnumAttr<"ActKind", "Eclipse activation kind",
    [Eclipse_ActKind_RELU, Eclipse_ActKind_EXP, Eclipse_ActKind_RSQRT,
     Eclipse_ActKind_SILU]> {
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

class Eclipse_EwiseOp<string mnemonic>
    : Eclipse_Op<mnemonic> {
  let arguments = (ins Eclipse_SRAMMemRef:$lhs, Eclipse_SRAMMemRef:$rhs,
                       Eclipse_SRAMMemRef:$dst);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $dst attr-dict `:` type($lhs) `,` type($rhs) `,` type($dst)";
}

def Eclipse_EwiseAddOp : Eclipse_EwiseOp<"elementwise_add">;
def Eclipse_EwiseSubOp : Eclipse_EwiseOp<"elementwise_sub">;
def Eclipse_EwiseMulOp : Eclipse_EwiseOp<"elementwise_mul">;
def Eclipse_EwiseDivOp : Eclipse_EwiseOp<"elementwise_div">;

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

def Eclipse_ReduceKind_MAX : I32EnumAttrCase<"MAX", 0, "max">;
def Eclipse_ReduceKind_SUM : I32EnumAttrCase<"SUM", 1, "sum">;
def Eclipse_ReduceKind_SQUARE_SUM : I32EnumAttrCase<"SQUARE_SUM", 2, "square_sum">;
def Eclipse_ReduceKind_ARGMAX : I32EnumAttrCase<"ARGMAX", 3, "argmax">;
def Eclipse_ReduceKindAttr : I32EnumAttr<"ReduceKind", "Eclipse reduce kind",
    [Eclipse_ReduceKind_MAX, Eclipse_ReduceKind_SUM, Eclipse_ReduceKind_SQUARE_SUM,
     Eclipse_ReduceKind_ARGMAX]> {
  let cppNamespace = "::mlir::eclipse";
}

// REDUCE_ARGMAX 吐出来的索引是 u32，所以要一个 i32 的 SRAM 类型
def Eclipse_SRAMIndexMemRef : ConfinedType<
    MemRefRankOf<[I32], [2]>,
    [CPred<"llvm::cast<::mlir::MemRefType>($_self).getMemorySpaceAsInt() == 0">],
    "2D i32 memref in SRAM", "::mlir::MemRefType">;

def Eclipse_ReduceOp : Eclipse_Op<"reduce"> {
  let summary = "Eclipse NPU row-wise reduction";
  let description = [{
    `eclipse.reduce` corresponds to the REDUCE instruction in the ISA.
    It folds every row of `src` ([rows, cols]) into one value,
    `dst[r, 0] = fold(src[r, 0..cols))`, so `dst` is [rows, 1].
    `kind` selects the fold: max / sum / square_sum / argmax.
    `kind = argmax` additionally writes the index of the first maximum into
    `idx` ([rows, 1], i32); the other kinds must not carry `idx`.
  }];

  let arguments = (ins
    Eclipse_SRAMMemRef:$src,
    Eclipse_SRAMMemRef:$dst,
    Optional<Eclipse_SRAMIndexMemRef>:$idx,
    Eclipse_ReduceKindAttr:$kind
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = "$src `,` $dst (`,` $idx^)? `{` `kind` `=` $kind `}` attr-dict `:` type($src) `,` type($dst) (`,` type($idx)^)?";
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

