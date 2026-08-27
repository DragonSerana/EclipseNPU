#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

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
        auto castOp = dmaloadOp.getSrc().getDefiningOp<memref::MemorySpaceCastOp>();
        auto arg = castOp.getSource();
        llvm::errs() << "ly @@@@@@@@@@ arg = " << arg << "\n";
      }
    });
  }
};

} // namespace

} // namespace mlir::eclipse
