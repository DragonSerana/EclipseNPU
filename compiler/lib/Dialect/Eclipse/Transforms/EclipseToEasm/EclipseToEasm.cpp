#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <system_error>
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Format.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

namespace {

void emitDmaLoadOp(DmaLoadOp dmaloadOp, llvm::raw_ostream &fileOS) {
  auto castOp =
      dmaloadOp.getSrc().getDefiningOp<memref::MemorySpaceCastOp>();
  auto arg = castOp.getSource();
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(arg)) {
    auto funcOp =
        mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    auto addrAttr =
        funcOp.getArgAttr(blockArg.getArgNumber(), "eclipse.ddr_addr");
    uint32_t addr =
        static_cast<uint32_t>(mlir::cast<IntegerAttr>(addrAttr).getInt());
    
    fileOS << llvm::formatv("{0,-15} ddr=0x{1:x}\n", "DMA_LOAD", addr);
    // DMA_LOAD        desc=0x80000100 sram=0x10000000 ddr=0x80010000 rows=16 cols=16 srcStride=32 dstStride=32    
  }  
}

class EclipseToEasm : public impl::EclipseToEasmBase<EclipseToEasm> {
public:
  using impl::EclipseToEasmBase<EclipseToEasm>::EclipseToEasmBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    std::error_code EC;
    llvm::raw_fd_ostream fileOS(outputFileName, EC, llvm::sys::fs::OF_None);
    if(EC) {
      module.emitError("Failed to open file ") << outputFileName << 
        ".Error " << EC.message();
      signalPassFailure();
      return;
    }

    module->walk([&](Operation *op) {
      if (auto dmaloadOp = mlir::dyn_cast<DmaLoadOp>(op)) {
        emitDmaLoadOp(dmaloadOp, fileOS);
      }
      if (auto dmastoreOp = mlir::dyn_cast<DmaStoreOp>(op)) {
        auto castOp =
            dmastoreOp.getDst().getDefiningOp<memref::MemorySpaceCastOp>();
        auto dst = castOp.getSource();
        auto addrAttr = dst.getDefiningOp<memref::AllocOp>()->getAttrOfType<IntegerAttr>("eclipse.ddr_addr");
        uint32_t addr = addrAttr.getValue().getZExtValue();
        llvm::errs() << "ly @@@@@@@@@ result addr = " << addr << "\n";
      }
    });
  }
};

} // namespace

} // namespace mlir::eclipse
