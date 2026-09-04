#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h.inc"
#include "eclipse/Dialect/Eclipse/Transforms/ElideCopiesPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSEELIDECOPIES
#include "EclipseElideCopiesPasses.h.inc"

namespace {

Value getViewSource(Value v) {                          
  while (auto *defOp = v.getDefiningOp()) {            
    if (auto subViewOp = dyn_cast<memref::SubViewOp>(defOp))
      v = subViewOp.getSource();
    else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(defOp))  
      v = castOp.getSource();
    else
      break;
  }
  return v;
}

class EclipseElideCopies
    : public impl::EclipseElideCopiesBase<EclipseElideCopies> {
public:
  using impl::EclipseElideCopiesBase<EclipseElideCopies>::EclipseElideCopiesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module->walk([&](DmaLoadOp damLoadOp){

    });
  }
};

} // namespace

} // namespace mlir::eclipse
