#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

uint32_t emitSyncOp(Operation *op, llvm::raw_ostream &fileOS,
                    uint32_t descAddr) {
  fileOS << "SYNC\n";
  return descAddr;
}

} // namespace mlir::eclipse
