#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult MatmulOp::verify() {
  // TODO(user): 填写 matmul 的结构检查。
  return success();
}
