#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult MatmulOp::verify() {
  auto lhsType = mlir::dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(getRhs().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!lhsType || !rhsType || !dstType)
    return emitOpError("MatmulOp operands must be MemRef types");

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto dstShape = dstType.getShape();

  if (lhsShape[1] != rhsShape[0])
    return emitOpError("MatmulOp lhs K must match rhs M (")
           << lhsShape[1] << " vs " << rhsShape[0] << ")";

  if (dstShape[0] != lhsShape[0])
    return emitOpError("MatmulOp dst M must match lhs M (")
           << dstShape[0] << " vs " << lhsShape[0] << ")";

  if (dstShape[1] != rhsShape[1])
    return emitOpError("MatmulOp dst N must match rhs N (")
           << dstShape[1] << " vs " << rhsShape[1] << ")";

  return success();
}
