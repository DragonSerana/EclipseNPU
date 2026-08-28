#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir::eclipse {

static constexpr uint32_t DESC_STRIDE = 0x40;  

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

namespace {

uint32_t getDdrAddress(Value value, func::FuncOp funcOp) {
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
    auto attr = funcOp.getArgAttr(blockArg.getArgNumber(), "eclipse.ddr_addr");
    if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(attr))
      return static_cast<uint32_t>(intAttr.getInt());
    return 0;
  }

  if (auto cast = value.getDefiningOp<memref::MemorySpaceCastOp>())
    return getDdrAddress(cast.getSource(), funcOp);

  if (auto alloc = value.getDefiningOp<memref::AllocOp>()) {
    auto attr = alloc->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");
    if (attr)
      return static_cast<uint32_t>(attr.getInt());
  }

  return 0;
}

uint32_t getSramAddress(Value value) {
  if (auto sram = value.getDefiningOp<SramOp>())
    return sram.getAddr();
  return 0;
}

uint32_t getElementByteSize(MemRefType type) {
  return type.getElementTypeBitWidth() / 8;
}

uint32_t getRowBytes(MemRefType type) {
  auto shape = type.getShape();
  if (shape.size() < 2)
    return 0;
  return static_cast<uint32_t>(shape[1]) * getElementByteSize(type);
}


uint32_t getRowStrideBytes(MemRefType type) {
  auto [strides, offset] = type.getStridesAndOffset();
  if (strides.empty())
    return 0;
  if (strides[0] == ShapedType::kDynamic)
    return getRowBytes(type);
  return static_cast<uint32_t>(strides[0]) * getElementByteSize(type);
}

void emitDmaLoad(llvm::raw_ostream &os, DmaLoadOp op, func::FuncOp funcOp,
                 uint32_t &descPtr) {
  Value src = op.getSrc();
  Value dst = op.getDst();
  auto srcType = mlir::cast<MemRefType>(src.getType());
  uint32_t rows = static_cast<uint32_t>(srcType.getShape()[0]);
  uint32_t cols = static_cast<uint32_t>(srcType.getShape()[1]);
  uint32_t srcStride = getRowStrideBytes(srcType);
  auto dstType = mlir::cast<MemRefType>(dst.getType());
  uint32_t dstStride = cols * getElementByteSize(dstType);

  os << llvm::format("DMA_LOAD        desc=0x%08x ", descPtr);
  os << llvm::format("sram=0x%08x ddr=0x%08x ", getSramAddress(dst),
                     getDdrAddress(src, funcOp));
  os << llvm::format("rows=%u cols=%u ", rows, cols);
  os << llvm::format("srcStride=%u dstStride=%u\n", srcStride, dstStride);
  descPtr += DESC_STRIDE;
}

void emitDmaStore(llvm::raw_ostream &os, DmaStoreOp op, func::FuncOp funcOp,
                  uint32_t &descPtr) {
  Value src = op.getSrc();
  Value dst = op.getDst();
  auto srcType = mlir::cast<MemRefType>(src.getType());
  uint32_t rows = static_cast<uint32_t>(srcType.getShape()[0]);
  uint32_t cols = static_cast<uint32_t>(srcType.getShape()[1]);
  uint32_t srcStride = cols * getElementByteSize(srcType);
  auto dstType = mlir::cast<MemRefType>(dst.getType());
  uint32_t dstStride = getRowStrideBytes(dstType);

  os << llvm::format("DMA_STORE       desc=0x%08x ", descPtr);
  os << llvm::format("sram=0x%08x ddr=0x%08x ", getSramAddress(src),
                     getDdrAddress(dst, funcOp));
  os << llvm::format("rows=%u cols=%u ", rows, cols);
  os << llvm::format("srcStride=%u dstStride=%u\n", srcStride, dstStride);
  descPtr += DESC_STRIDE;
}

void emitSync(llvm::raw_ostream &os, SyncOp op) { os << "SYNC\n"; }

void emitMatmul(llvm::raw_ostream &os, MatmulOp op, func::FuncOp funcOp,
                uint32_t &descPtr) {
  // TODO
}

void emitEwiseAdd(llvm::raw_ostream &os, EwiseAddOp op, func::FuncOp funcOp,
                  uint32_t &descPtr) {
  // TODO
}

void emitAct(llvm::raw_ostream &os, ActOp op, func::FuncOp funcOp,
             uint32_t &descPtr) {
  // TODO
}

class EclipseToEasm : public impl::EclipseToEasmBase<EclipseToEasm> {
public:
  using impl::EclipseToEasmBase<EclipseToEasm>::EclipseToEasmBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string easmText;
    llvm::raw_string_ostream os(easmText);
    uint32_t descPtr = 0x80000100;

    module->walk([&](Operation *op) {
      if (auto dmaLoad = mlir::dyn_cast<DmaLoadOp>(op)) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        emitDmaLoad(os, dmaLoad, funcOp, descPtr);
      } else if (auto dmaStore = mlir::dyn_cast<DmaStoreOp>(op)) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        emitDmaStore(os, dmaStore, funcOp, descPtr);
      } else if (auto sync = mlir::dyn_cast<SyncOp>(op)) {
        emitSync(os, sync);
      } else if (auto matmul = mlir::dyn_cast<MatmulOp>(op)) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        emitMatmul(os, matmul, funcOp, descPtr);
      } else if (auto add = mlir::dyn_cast<EwiseAddOp>(op)) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        emitEwiseAdd(os, add, funcOp, descPtr);
      } else if (auto act = mlir::dyn_cast<ActOp>(op)) {
        auto funcOp = op->getParentOfType<func::FuncOp>();
        emitAct(os, act, funcOp, descPtr);
      }
    });

    os.flush();

    if (!output.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream file(output, ec);
      if (ec) {
        module.emitError() << "cannot open output file: " << ec.message();
        signalPassFailure();
        return;
      }
      file << easmText;
    } else {
      llvm::outs() << easmText;
    }
  }
};

} // namespace

} // namespace mlir::eclipse
