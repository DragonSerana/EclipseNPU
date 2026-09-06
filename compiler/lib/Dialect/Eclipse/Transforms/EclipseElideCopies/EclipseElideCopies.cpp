#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h.inc"
#include "eclipse/Dialect/Eclipse/Transforms/ElideCopiesPasses.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSEELIDECOPIES
#include "EclipseElideCopiesPasses.h.inc"

namespace {
struct Match {
  DmaStoreOp storeOp;
  DmaLoadOp loadOp;
};

Value getCastSource(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(defOp))
      v = castOp.getSource();
    else
      break;
  }
  return v;
}

bool viewOnlyFeedsStore(Operation *viewOp, DmaStoreOp storeOp) {
  for (auto &vUse : viewOp->getResult(0).getUses()) {
    Operation *v = vUse.getOwner();
    if (v == storeOp.getOperation())
      continue;
    if (isa<memref::MemorySpaceCastOp, memref::SubViewOp>(v))
      continue;
    return false;
  }
  return true;
}

bool rootIsDead(DmaStoreOp storeOp) {
  auto base = getCastSource(storeOp.getDst());

  for (auto const &user : base.getUsers()) {
    // 获取storeOp的存放DDR的value，找到他的使用者，必须是subview/castop
    if (mlir::isa<memref::SubViewOp, memref::MemorySpaceCastOp>(user)) {
      // 递归判断这个 value只能被store消费，穿透subview/castop搜索
      if (!viewOnlyFeedsStore(user, storeOp))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

class EclipseElideCopies
    : public impl::EclipseElideCopiesBase<EclipseElideCopies> {
public:
  using impl::EclipseElideCopiesBase<
      EclipseElideCopies>::EclipseElideCopiesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<Match, 4> matchList;
    DenseMap<Value, DmaStoreOp> lastStore;

    module->walk([&](Operation *op) {
      if (auto storeOp = dyn_cast<DmaStoreOp>(op)) {
        Value storeKey = getCastSource(storeOp.getDst());
        lastStore[storeKey] = storeOp;
      } else if (auto loadOp = dyn_cast<DmaLoadOp>(op)) {
        Value loadKey = getCastSource(loadOp.getSrc());
        auto it = lastStore.find(loadKey);
        if (it == lastStore.end())
          return;

        auto storeOp = it->second;
        if (storeOp->getBlock() != loadOp->getBlock())
          return;

        matchList.push_back({storeOp, loadOp});
      }
    });

    for (auto &m : matchList) {
      m.loadOp.getDst().replaceAllUsesWith(m.storeOp.getSrc());
      m.loadOp.erase();
    }

    DenseSet<DmaStoreOp> done;
    for (auto &m : matchList) {
      if (!done.insert(m.storeOp).second)
        continue;

      if (rootIsDead(m.storeOp))
        m.storeOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::eclipse
