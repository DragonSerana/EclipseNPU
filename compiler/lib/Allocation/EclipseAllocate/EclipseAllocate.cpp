#include "eclipse/Allocation/Passes.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSEALLOCATE
#include "EclipseAllocationPasses.h.inc"

namespace {

class EclipseAllocate : public impl::EclipseAllocateBase<EclipseAllocate> {
public:
  using impl::EclipseAllocateBase<EclipseAllocate>::EclipseAllocateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module->walk([&](memref::AllocOp alloc){
      auto memType = mlir::cast<MemRefType>(alloc.getType());
      auto memSpace = memType.getMemorySpaceAsInt();

      bool isSRAM = true;
      if(memSpace != 0) 
        isSRAM = false;

      for (auto use : alloc->getResults().getUsers()) {
        if(auto castOp = mlir::dyn_cast<memref::MemorySpaceCastOp>(use)) {
          auto dstType = mlir::cast<MemRefType>(castOp.getType());
          if (dstType.getMemorySpaceAsInt() == 1)
            isSRAM = false;
        }
      }
      
      if(isSRAM) {
        llvm::errs() << "Ly @@@@@@ alloc SRAM @@@@@@@\n";
      }
    });
    getContext().getOrLoadDialect<EclipseDialect>();
  }
};

} // namespace

} // namespace mlir::eclipse
