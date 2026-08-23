#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult EwiseAddOp::verify() {
  // TODO(user): 填写 elementwise_add 的结构检查。
  return success();
}
