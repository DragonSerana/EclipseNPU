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
        emitSyncOp(syncOp, fileOS);
      } else if (auto matmulOp = mlir::dyn_cast<MatmulOp>(op)) {
        emitMatmulOp(matmulOp, fileOS);
      } else if (auto ewiseOp = mlir::dyn_cast<EwiseAddOp>(op)) {
        emitEwiseAddOp(ewiseOp, fileOS);
      } else if (auto actOp = mlir::dyn_cast<ActOp>(op)) {
        emitActOp(actOp, fileOS);
      }
    });
  }

private:
  void emitDmaLoadOp(DmaLoadOp dmaloadOp, llvm::raw_ostream &fileOS) {
    auto castOp = dmaloadOp.getSrc().getDefiningOp<memref::MemorySpaceCastOp>();
    auto arg = castOp.getSource();

    uint32_t addr = 0;
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(arg)) {
      auto funcOp =
          mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
      auto addrAttr =
          funcOp.getArgAttr(blockArg.getArgNumber(), "eclipse.ddr_addr");
      addr = static_cast<uint32_t>(mlir::cast<IntegerAttr>(addrAttr).getInt());
    } else if (auto allocOp = arg.getDefiningOp<memref::AllocOp>()) {
      auto addrAttr = allocOp->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");
      addr = static_cast<uint32_t>(addrAttr.getValue().getZExtValue());
    } else {
      return;
    }

    uint32_t sram = dmaloadOp.getDst().getDefiningOp<SramOp>().getAddr();

    auto rows = arg.getType().getShape()[0];
    auto cols = arg.getType().getShape()[1];
    // TODO 目前stride只有packed
    auto srcStride = cols * eclipse_runtime::DTYPE_SIZE;
    auto dstStride = cols * eclipse_runtime::DTYPE_SIZE;

    fileOS << llvm::formatv("{0,-15} desc={1:x} sram={2:x} ddr={3:x} "
                            "rows={4:d} cols={5:d} srcStride={6:d} "
                            "dstStride={7:d}\n",
                            "DMA_LOAD", descAddr_, sram, addr, rows, cols,
                            srcStride, dstStride);

    descAddr_ += DESC_LEN;
  }

  void emitDmaStoreOp(DmaStoreOp dmastoreOp, llvm::raw_ostream &fileOS) {
    auto castOp =
        dmastoreOp.getDst().getDefiningOp<memref::MemorySpaceCastOp>();
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
        "DMA_STORE", descAddr_, sram, addr, rows, cols, srcStride, dstStride);

    descAddr_ += DESC_LEN;
  }

  void emitSyncOp(SyncOp syncOp, llvm::raw_ostream &fileOS) {
    fileOS << "SYNC\n";
  }

  void emitMatmulOp(MatmulOp matmulOp, llvm::raw_ostream &fileOS) {
    uint32_t dst = matmulOp.getDst().getDefiningOp<SramOp>().getAddr();
    uint32_t lhs = matmulOp.getLhs().getDefiningOp<SramOp>().getAddr();
    uint32_t rhs = matmulOp.getRhs().getDefiningOp<SramOp>().getAddr();

    //TODO Matmul参数暂时写死
    uint32_t M = matmulOp.getLhs().getType().getShape()[0];
    uint32_t K = matmulOp.getLhs().getType().getShape()[1];
    uint32_t N = matmulOp.getRhs().getType().getShape()[1];
    uint32_t acc = matmulOp.getAccumulate();
    fileOS << llvm::formatv(
        "{0,-15} desc={1:x} dst={2:x} lhs={3:x} rhs={4:x} "
        "M={5:d} N={6:d} K={7:d} acc={8:d}\n",
        "MATMUL", descAddr_, dst, lhs, rhs, M, N, K, acc);

    descAddr_ += DESC_LEN;

  }  

  void emitEwiseAddOp(EwiseAddOp ewiseOp, llvm::raw_ostream &fileOS) {
    uint32_t dst = ewiseOp.getDst().getDefiningOp<SramOp>().getAddr();
    uint32_t lhs = ewiseOp.getLhs().getDefiningOp<SramOp>().getAddr();
    uint32_t rhs = ewiseOp.getRhs().getDefiningOp<SramOp>().getAddr();

    uint32_t n = 1;
    for (auto dim : ewiseOp.getLhs().getType().getShape())
      n *= dim;

    fileOS << llvm::formatv(
        "{0,-15} desc={1:x} dst={2:x} lhs={3:x} rhs={4:x} n={5:d}\n",
        "ELEMENTWISE_ADD", descAddr_, dst, lhs, rhs, n);

    descAddr_ += DESC_LEN;
  }

  void emitActOp(ActOp actOp, llvm::raw_ostream &fileOS) {
    uint32_t dst = actOp.getDst().getDefiningOp<SramOp>().getAddr();
    uint32_t src = actOp.getSrc().getDefiningOp<SramOp>().getAddr();

    uint32_t n = 1;
    for (auto dim : actOp.getSrc().getType().getShape())
      n *= dim;

    uint32_t kind = static_cast<uint32_t>(actOp.getKind());

    fileOS << llvm::formatv(
        "{0,-15} desc={1:x} dst={2:x} src={3:x} n={4:d} kind={5:d}\n",
        "ACT", descAddr_, dst, src, n, kind);

    descAddr_ += DESC_LEN;
  }

  uint32_t descAddr_ = DESC_STARTADDR;
};

} // namespace

} // namespace mlir::eclipse
