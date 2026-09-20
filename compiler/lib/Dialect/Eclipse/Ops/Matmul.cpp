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

  // 转置时 lhs 按 [K,M]、rhs 按 [N,K] 摆布，逻辑上仍算 MxK 和 KxN
  const int64_t m = getTransA() ? lhsShape[1] : lhsShape[0];
  const int64_t lhsK = getTransA() ? lhsShape[0] : lhsShape[1];
  const int64_t rhsK = getTransB() ? rhsShape[1] : rhsShape[0];
  const int64_t n = getTransB() ? rhsShape[0] : rhsShape[1];

  if (lhsK != rhsK)
    return emitOpError("MatmulOp lhs K must match rhs M (")
           << lhsK << " vs " << rhsK << ")";

  if (dstShape[0] != m)
    return emitOpError("MatmulOp dst M must match lhs M (")
           << dstShape[0] << " vs " << m << ")";

  if (dstShape[1] != n)
    return emitOpError("MatmulOp dst N must match rhs N (")
           << dstShape[1] << " vs " << n << ")";

  return success();
}
