#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::eclipse;

namespace {

/// 四个 elementwise op 的形状约束完全一样：lhs/rhs/dst 都是 SRAM memref 且同形状。
LogicalResult verifyEwiseShapes(Operation *op, Value lhs, Value rhs, Value dst) {
  auto lhsType = mlir::dyn_cast<MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(rhs.getType());
  auto dstType = mlir::dyn_cast<MemRefType>(dst.getType());

  if (!lhsType || !rhsType || !dstType)
    return op->emitOpError("operands must be MemRef types");

  if (lhsType.getShape() != rhsType.getShape())
    return op->emitOpError("lhs and rhs shapes must match (")
           << lhsType.getShape() << " vs " << rhsType.getShape() << ")";

  if (lhsType.getShape() != dstType.getShape())
    return op->emitOpError("lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  return success();
}

} // namespace

LogicalResult EwiseAddOp::verify() {
  return verifyEwiseShapes(getOperation(), getLhs(), getRhs(), getDst());
}

LogicalResult EwiseSubOp::verify() {
  return verifyEwiseShapes(getOperation(), getLhs(), getRhs(), getDst());
}

LogicalResult EwiseMulOp::verify() {
  return verifyEwiseShapes(getOperation(), getLhs(), getRhs(), getDst());
}

LogicalResult EwiseDivOp::verify() {
  return verifyEwiseShapes(getOperation(), getLhs(), getRhs(), getDst());
}
