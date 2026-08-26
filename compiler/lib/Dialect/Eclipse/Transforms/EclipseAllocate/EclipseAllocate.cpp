#include "eclipse/Dialect/Eclipse/EclipseConstants.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/Transforms/AllocationPasses.h"

#include "SramAllocator.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "runtime/include/eclipse_isa.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSEALLOCATE
#include "EclipseAllocationPasses.h.inc"

namespace {

class EclipseAllocate : public impl::EclipseAllocateBase<EclipseAllocate> {
public:
  using impl::EclipseAllocateBase<EclipseAllocate>::EclipseAllocateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SramAllocator sramAlloc(::eclipse_runtime::SRAM_ADDR,
                            ::eclipse_runtime::SRAM_SIZE);

    // TODO.
    // ddr地址alloc,暂时先写死，这里为了避开命令队列区，在mamtmul_golden.cpp的CMD_BASE宏
    uint32_t curDdrAddr = ::eclipse_runtime::DDR_ADDR + 0x10000;
    const uint32_t ddrStride = 0x10000;

    OpBuilder builder_ctx(&getContext());
    module.walk([&](func::FuncOp funcOp) {
      for (uint32_t i = 0; i < funcOp.getNumArguments(); i++) {
        auto argType = funcOp.getArgumentTypes()[i];
        if (!mlir::dyn_cast<MemRefType>(argType))
          continue;

        auto addrAttr = builder_ctx.getI64IntegerAttr(curDdrAddr);
        funcOp.setArgAttr(i, "eclipse.ddr_addr", addrAttr);
        curDdrAddr += ddrStride;
      }

      for (uint32_t i = 0; i < funcOp.getNumResults(); i++) {
        auto resultType = funcOp.getResultTypes()[i];
        if (!mlir::dyn_cast<MemRefType>(resultType))
          continue;

        auto addrAttr = builder_ctx.getI64IntegerAttr(curDdrAddr);
        funcOp.setResultAttr(i, "eclipse.ddr_addr", addrAttr);
        curDdrAddr += ddrStride;
      }

      if (curDdrAddr >
          ::eclipse_runtime::DDR_ADDR + ::eclipse_runtime::DDR_SIZE) {
        funcOp->emitError("DDR allocation failed: out of DDR");
        signalPassFailure();
        return;
      }
    });

    module->walk([&](memref::AllocOp alloc) {
      auto memType = mlir::cast<MemRefType>(alloc.getType());
      auto memSpace = memType.getMemorySpaceAsInt();

      bool isSRAM = true;
      if (memSpace != SRAM_MEMORY_SPACE)
        isSRAM = false;

      for (auto use : alloc->getResults().getUsers()) {
        if (auto castOp = mlir::dyn_cast<memref::MemorySpaceCastOp>(use)) {
          auto dstType = mlir::cast<MemRefType>(castOp.getType());
          if (dstType.getMemorySpaceAsInt() == DDR_MEMORY_SPACE)
            isSRAM = false;
        }
      }

      if (isSRAM) {
        auto allocNumElement = alloc.getType().getNumElements();
        auto size =
            allocNumElement * alloc.getType().getElementTypeBitWidth() / 8;

        uint64_t alignment = SRAM_ALIGNMENT;
        if (auto alignAttr = alloc.getAlignment()) {
          alignment = alignAttr.value();
        }

        uint32_t addr = sramAlloc.allocate(static_cast<uint32_t>(size),
                                           static_cast<uint32_t>(alignment));
        if (addr == 0) {
          alloc->emitError("SRAM allocation failed: out of SRAM");
          signalPassFailure();
          return;
        }

        OpBuilder builder(alloc);
        auto sramOp =
            SramOp::create(builder, alloc->getLoc(), alloc.getType(), addr);
        alloc.getResult().replaceAllUsesWith(sramOp.getBuffer());
        alloc.erase();
      }
    });
  }
};

} // namespace

} // namespace mlir::eclipse
