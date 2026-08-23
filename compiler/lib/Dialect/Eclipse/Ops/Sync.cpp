#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult SyncOp::verify() {
  // SYNC 无操作数、无属性，v0.1 无需额外检查。
  return success();
}
