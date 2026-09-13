#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "runtime/include/eclipse_isa.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

namespace {

/// 取 DMA 源/目的在 DDR 里的物理地址：既可能是函数实参（带 eclipse.ddr_addr
/// 属性），也可能是带该属性的 alloc。
uint32_t ddrBaseAddr(Value base) {
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(base)) {
    auto funcOp = mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    auto addrAttr =
        funcOp.getArgAttr(blockArg.getArgNumber(), "eclipse.ddr_addr");
    return static_cast<uint32_t>(mlir::cast<IntegerAttr>(addrAttr).getInt());
  }
  if (auto allocOp = base.getDefiningOp<memref::AllocOp>()) {
    auto addrAttr = allocOp->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");
    return static_cast<uint32_t>(addrAttr.getValue().getZExtValue());
  }
  return 0;
}

uint32_t emitDmaLoadOp(DmaLoadOp dmaloadOp, llvm::raw_ostream &fileOS,
                       uint32_t descAddr) {
  auto src = dmaloadOp.getSrc();
  uint32_t sram = dmaloadOp.getDst().getDefiningOp<SramOp>().getAddr();

  if (auto castOp =
          llvm::dyn_cast<memref::MemorySpaceCastOp>(src.getDefiningOp())) {
    auto arg = castOp.getSource();
    if (!mlir::isa<BlockArgument>(arg) &&
        !arg.getDefiningOp<memref::AllocOp>())
      return descAddr;
    uint32_t addr = ddrBaseAddr(arg);

    auto rows = arg.getType().getShape()[0];
    auto cols = arg.getType().getShape()[1];
    // TODO 目前stride只有packed
    auto srcStride = cols * eclipse_runtime::DTYPE_SIZE;
    auto dstStride = cols * eclipse_runtime::DTYPE_SIZE;

    fileOS << llvm::formatv("{0,-15} desc={1:x} sram={2:x} ddr={3:x} "
                            "rows={4:d} cols={5:d} srcStride={6:d} "
                            "dstStride={7:d}\n",
                            "DMA_LOAD", descAddr, sram, addr, rows, cols,
                            srcStride, dstStride);

    return descAddr + DESC_LEN;
  }

  if (auto subViewOp =
          llvm::dyn_cast<memref::SubViewOp>(src.getDefiningOp())) {
    llvm::SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();
    llvm::SmallVector<OpFoldResult> sizes = subViewOp.getMixedSizes();
    llvm::SmallVector<OpFoldResult> strides = subViewOp.getMixedStrides();

    auto castOp =
        subViewOp.getSource().getDefiningOp<memref::MemorySpaceCastOp>();
    auto arg = castOp.getSource();
    if (!mlir::isa<BlockArgument>(arg) &&
        !arg.getDefiningOp<memref::AllocOp>())
      return descAddr;
    uint32_t baseAddr = ddrBaseAddr(arg);

    auto getV = [](OpFoldResult f) -> int64_t {
      if (auto v = getConstantIntValue(f))
        return *v;
      return mlir::cast<arith::ConstantIndexOp>(
                 mlir::cast<Value>(f).getDefiningOp())
          .value();
    };

    uint32_t rowOffset = getV(offsets[0]);
    uint32_t colOffset = getV(offsets[1]);
    uint32_t rows = getV(sizes[0]);
    uint32_t cols = getV(sizes[1]);
    // 内存行 stride = base 的列数(packed)
    auto baseType = mlir::cast<MemRefType>(arg.getType());
    uint32_t rowSrcStride =
        baseType.getShape()[1] * eclipse_runtime::DTYPE_SIZE;
    uint32_t colStride = getV(strides[1]) * eclipse_runtime::DTYPE_SIZE;
    // 这里应该用切分的size,因为放到sram是切分后的
    uint32_t rowDstStride = getV(sizes[1]) * eclipse_runtime::DTYPE_SIZE;

    auto addr = baseAddr + rowOffset * rowSrcStride + colOffset * colStride;

    fileOS << llvm::formatv("{0,-15} desc={1:x} sram={2:x} ddr={3:x} "
                            "rows={4:d} cols={5:d} srcStride={6:d} "
                            "dstStride={7:d}\n",
                            "DMA_LOAD", descAddr, sram, addr, rows, cols,
                            rowSrcStride, rowDstStride);

    return descAddr + DESC_LEN;
  }

  llvm_unreachable("dmaloadOp src getDefiningOp must be Cast/Subview");
}

uint32_t emitDmaStoreOp(DmaStoreOp dmastoreOp, llvm::raw_ostream &fileOS,
                        uint32_t descAddr) {
  auto castOp = dmastoreOp.getDst().getDefiningOp<memref::MemorySpaceCastOp>();
  auto allocOp = castOp.getSource().getDefiningOp<memref::AllocOp>();

  auto addrAttr = allocOp->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");
  uint32_t addr = static_cast<uint32_t>(addrAttr.getValue().getZExtValue());

  uint32_t sram = dmastoreOp.getSrc().getDefiningOp<SramOp>().getAddr();

  auto memrefType = mlir::cast<MemRefType>(allocOp.getType());
  auto rows = memrefType.getShape()[0];
  auto cols = memrefType.getShape()[1];
  // TODO 目前 stride 只有 packed
  auto srcStride = cols * eclipse_runtime::DTYPE_SIZE;
  auto dstStride = cols * eclipse_runtime::DTYPE_SIZE;

  fileOS << llvm::formatv(
      "{0,-15} desc={1:x} sram={2:x} ddr={3:x} rows={4:d} cols={5:d} "
      "srcStride={6:d} dstStride={7:d}\n",
      "DMA_STORE", descAddr, sram, addr, rows, cols, srcStride, dstStride);

  return descAddr + DESC_LEN;
}

} // namespace

uint32_t emitDmaOp(Operation *op, llvm::raw_ostream &fileOS,
                   uint32_t descAddr) {
  if (auto dmaloadOp = mlir::dyn_cast<DmaLoadOp>(op))
    return emitDmaLoadOp(dmaloadOp, fileOS, descAddr);
  if (auto dmastoreOp = mlir::dyn_cast<DmaStoreOp>(op))
    return emitDmaStoreOp(dmastoreOp, fileOS, descAddr);
  llvm_unreachable("emitDmaOp: not a DMA op");
}

} // namespace mlir::eclipse
