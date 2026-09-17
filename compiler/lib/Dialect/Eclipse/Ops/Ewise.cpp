#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::eclipse;

namespace {

/// 四个 elementwise op 的形状规则一样：lhs/dst 同形、都是二维 [rows, cols]；
/// rhs 是 [rows', cols']，rows' 为 1 或 rows，cols' 整除 cols。rhs 的读模式
/// 由描述符的 cols/rhsBlk/rhsStride 描述，见 docs/spec/isa.md。
LogicalResult verifyEwiseShapes(Operation *op, Value lhs, Value rhs,
                                Value dst) {
  auto lhsType = mlir::dyn_cast<MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<MemRefType>(rhs.getType());
  auto dstType = mlir::dyn_cast<MemRefType>(dst.getType());

  if (!lhsType || !rhsType || !dstType)
    return op->emitOpError("operands must be MemRef types");

  if (lhsType.getShape() != dstType.getShape())
    return op->emitOpError("lhs and dst shapes must match (")
           << lhsType.getShape() << " vs " << dstType.getShape() << ")";

  if (dstType.getRank() != 2 || rhsType.getRank() != 2)
    return op->emitOpError("dst and rhs must be rank-2 (")
           << dstType.getShape() << " vs " << rhsType.getShape() << ")";

  auto dstShape = dstType.getShape();
  auto rhsShape = rhsType.getShape();

  if (rhsShape[0] != 1 && rhsShape[0] != dstShape[0])
    return op->emitOpError("rhs rows must be 1 or match dst rows (")
           << rhsShape[0] << " vs " << dstShape[0] << ")";

  if (rhsShape[1] <= 0 || dstShape[1] % rhsShape[1] != 0)
    return op->emitOpError(
               "dst row width must be a multiple of rhs row width (")
           << dstShape[1] << " vs " << rhsShape[1] << ")";

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
