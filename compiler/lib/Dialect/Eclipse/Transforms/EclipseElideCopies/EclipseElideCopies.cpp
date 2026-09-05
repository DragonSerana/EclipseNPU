#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h.inc"
#include "eclipse/Dialect/Eclipse/Transforms/ElideCopiesPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"

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
    DenseMap<Value, DmaStoreOp> lastStore;

    module->walk([&](Operation *op) {
      if (auto storeOp = dyn_cast<DmaStoreOp>(op)) {
        Value storeKey = getViewSource(storeOp.getDst());
        lastStore[storeKey] = storeOp;
      } else if (auto loadOp = dyn_cast<DmaLoadOp>(op)) {
        Value loadKey = getViewSource(loadOp.getSrc());
        auto it = lastStore.find(loadKey);
      }
    });

  }
};

} // namespace

} // namespace mlir::eclipse
