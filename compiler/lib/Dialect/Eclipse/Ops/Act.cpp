#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult ActOp::verify() {
  auto srcType = mlir::dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!srcType || !dstType)
    return emitOpError("ActOp operands must be MemRef types");

  if (srcType.getShape() != dstType.getShape())
    return emitOpError("ActOp src and dst shapes must match (")
           << srcType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}
