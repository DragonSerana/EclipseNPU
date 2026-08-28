#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

namespace {

class EclipseToEasm : public impl::EclipseToEasmBase<EclipseToEasm> {
public:
  using impl::EclipseToEasmBase<EclipseToEasm>::EclipseToEasmBase;

  void runOnOperation() override {
    // TODO: 将 Eclipse IR 发射为 .easm 文本。
    ModuleOp module = getOperation();

    module->walk([&](Operation *op) {
      if (auto dmaloadOp = mlir::dyn_cast<DmaLoadOp>(op)) {
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
          llvm::errs() << "ly @@@@@@@@@ input addr = " << addr << "\n";
        }
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
