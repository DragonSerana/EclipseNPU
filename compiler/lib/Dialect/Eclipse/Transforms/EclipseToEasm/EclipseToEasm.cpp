#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "runtime/include/eclipse_isa.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <system_error>

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

constexpr uint32_t DESC_STARTADDR = 0x80000100;
constexpr uint32_t DESC_LEN = 0x40;

namespace {

class EclipseToEasm : public impl::EclipseToEasmBase<EclipseToEasm> {
public:
  using impl::EclipseToEasmBase<EclipseToEasm>::EclipseToEasmBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::error_code EC;
    llvm::raw_fd_ostream fileOS(outputFileName, EC, llvm::sys::fs::OF_None);
    if (EC) {
      module.emitError("Failed to open file ")
          << outputFileName << ".Error " << EC.message();
      signalPassFailure();
      return;
    }

    module->walk([&](Operation *op) {
      if (auto dmaloadOp = mlir::dyn_cast<DmaLoadOp>(op)) {
        emitDmaLoadOp(dmaloadOp, fileOS);
      } else if (auto dmastoreOp = mlir::dyn_cast<DmaStoreOp>(op)) {
        emitDmaStoreOp(dmastoreOp, fileOS);
      } else if (auto syncOp = mlir::dyn_cast<SyncOp>(op)) {
        fileOS << "SYNC\n";
      } else if (auto matmulOp = mlir::dyn_cast<MatmulOp>(op)) {
        
      }

      
    });
  }

private:
  void emitDmaLoadOp(DmaLoadOp dmaloadOp, llvm::raw_ostream &fileOS) {
    auto castOp = dmaloadOp.getSrc().getDefiningOp<memref::MemorySpaceCastOp>();
    auto arg = castOp.getSource();
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(arg)) {
      auto funcOp =
          mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
      auto addrAttr =
          funcOp.getArgAttr(blockArg.getArgNumber(), "eclipse.ddr_addr");
      uint32_t addr =
          static_cast<uint32_t>(mlir::cast<IntegerAttr>(addrAttr).getInt());

      auto rows = arg.getType().getShape()[0];
      auto cols = arg.getType().getShape()[1];
      // TODO 目前stride只有packed
      auto srcStride = cols * eclipse_runtime::DTYPE_SIZE;
      auto dstStride = cols * eclipse_runtime::DTYPE_SIZE;
      llvm::errs() << "ly @@@@@@@@ rows = " << rows << ", cols = " << cols
                   << "\n";
      descAddr_ += DESC_LEN;
      fileOS << llvm::formatv("{0,-15} desc=0x{1:x} ddr=0x{2:x} rows={3:d} "
                              "cols={4:d} srcStride={5:d} dstStride={6:d}\n",
                              "DMA_LOAD", descAddr_, addr, rows, cols,
                              srcStride, dstStride);
    }
  }

  void emitDmaStoreOp(DmaStoreOp dmastoreOp, llvm::raw_ostream &fileOS) {
    auto castOp =
        dmastoreOp.getDst().getDefiningOp<memref::MemorySpaceCastOp>();
    auto allocOp = castOp.getSource().getDefiningOp<memref::AllocOp>();

    auto addrAttr = allocOp->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");

    uint32_t addr = static_cast<uint32_t>(addrAttr.getValue().getZExtValue());

    auto memrefType = mlir::cast<MemRefType>(allocOp.getType());
    auto rows = memrefType.getShape()[0];
    auto cols = memrefType.getShape()[1];
    // TODO 目前 stride 只有 packed
    auto srcStride = cols * eclipse_runtime::DTYPE_SIZE;
    auto dstStride = cols * eclipse_runtime::DTYPE_SIZE;

    fileOS << llvm::formatv(
        "{0,-15} desc=0x{1:x} ddr=0x{2:x} rows={3:d} cols={4:d} "
        "srcStride={5:d} dstStride={6:d}\n",
        "DMA_STORE", descAddr_, addr, rows, cols, srcStride, dstStride);

    descAddr_ += DESC_LEN;
  }

  uint32_t descAddr_ = DESC_STARTADDR;
};

} // namespace

} // namespace mlir::eclipse
