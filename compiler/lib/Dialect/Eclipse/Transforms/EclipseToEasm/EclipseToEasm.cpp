#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "EmitHelpers.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <system_error>

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

constexpr uint32_t DESC_STARTADDR = 0x80000100;

namespace {

/// 把命令队列展开好的 Eclipse 指令按顺序发射成 .easm 文本。各 op 家族的具体
/// 发射在 *Emit.cpp 里，这里只做分发并维护 descriptor 地址。
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

    module.walk([&](scf::ForOp forOp) {
      if (failed(loopUnrollFull(forOp)))
        signalPassFailure();
    });

    mlir::OperationFolder folder(&getContext());
    module.walk([&](Operation *op) { (void)folder.tryToFold(op); });

    uint32_t descAddr = DESC_STARTADDR;
    module->walk([&](Operation *op) {
      if (isa<DmaLoadOp, DmaStoreOp>(op))
        descAddr = emitDmaOp(op, fileOS, descAddr);
      else if (isa<SyncOp>(op))
        descAddr = emitSyncOp(op, fileOS, descAddr);
      else if (isa<MatmulOp>(op))
        descAddr = emitMatmulOp(op, fileOS, descAddr);
      else if (isa<EwiseAddOp, EwiseSubOp, EwiseMulOp, EwiseDivOp>(op))
        descAddr = emitEwiseOp(op, fileOS, descAddr);
      else if (isa<ActOp>(op))
        descAddr = emitActOp(op, fileOS, descAddr);
    });
  }
};

} // namespace

} // namespace mlir::eclipse
