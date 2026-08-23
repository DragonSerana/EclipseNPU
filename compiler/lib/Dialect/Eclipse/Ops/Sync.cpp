#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult SyncOp::verify() {
  // TODO(user): 检查 SYNC 无操作数/无结果（如有必要）。
  return success();
}
