#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult EwiseAddOp::verify() {
  auto lhsType = mlir::dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(getRhs().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!lhsType || !rhsType || !dstType)
    return emitOpError("EwiseAddOp operands must be MemRef types");

  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("EwiseAddOp lhs and rhs shapes must match (")
           << lhsType.getShape() << " vs " << rhsType.getShape() << ")";

  if (lhsType.getShape() != dstType.getShape())
    return emitOpError("EwiseAddOp lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}

LogicalResult EwiseSubOp::verify() {
  auto lhsType = mlir::dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(getRhs().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!lhsType || !rhsType || !dstType)
    return emitOpError("EwiseSubOp operands must be MemRef types");

  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("EwiseSubOp lhs and rhs shapes must match (")
           << lhsType.getShape() << " vs " << rhsType.getShape() << ")";

  if (lhsType.getShape() != dstType.getShape())
    return emitOpError("EwiseSubOp lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}

LogicalResult EwiseMulOp::verify() {
  auto lhsType = mlir::dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(getRhs().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!lhsType || !rhsType || !dstType)
    return emitOpError("EwiseMulOp operands must be MemRef types");

  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("EwiseMulOp lhs and rhs shapes must match (")
           << lhsType.getShape() << " vs " << rhsType.getShape() << ")";

  if (lhsType.getShape() != dstType.getShape())
    return emitOpError("EwiseMulOp lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}

LogicalResult EwiseDivOp::verify() {
  auto lhsType = mlir::dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(getRhs().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!lhsType || !rhsType || !dstType)
    return emitOpError("EwiseDivOp operands must be MemRef types");

  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("EwiseDivOp lhs and rhs shapes must match (")
           << lhsType.getShape() << " vs " << rhsType.getShape() << ")";

  if (lhsType.getShape() != dstType.getShape())
    return emitOpError("EwiseDivOp lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}