#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult ReduceOp::verify() {
  auto srcType = mlir::dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!srcType || !dstType)
    return emitOpError("ReduceOp operands must be MemRef types");

  if (!srcType.hasStaticShape() || !dstType.hasStaticShape())
    return emitOpError("ReduceOp src/dst must have static shapes");

  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();

  if (srcShape[0] == 0 || srcShape[1] == 0)
    return emitOpError("ReduceOp src shape must be non-zero");

  // 归约方向由 cols 决定，输出恒为 [rows, 1]
  if (dstShape[0] != srcShape[0] || dstShape[1] != 1)
    return emitOpError("ReduceOp dst must be rows x 1, got ")
           << dstShape[0] << " x " << dstShape[1];

  const bool isArgmax = getKind() == ReduceKind::ARGMAX;
  if (isArgmax != (getIdx() != Value()))
    return emitOpError("ReduceOp idx is required by argmax, unused otherwise");

  if (isArgmax) {
    auto idxType = mlir::dyn_cast<MemRefType>(getIdx().getType());
    if (!idxType || !idxType.hasStaticShape())
      return emitOpError("ReduceOp idx must have a static shape");
    if (idxType.getShape() != dstShape)
      return emitOpError("ReduceOp idx must have the same shape as dst");
    if (getIdx() == getSrc() || getIdx() == getDst())
      return emitOpError("ReduceOp src/dst/idx must not alias");
  }

  // 与 MATMUL 同口径：保守禁止输入输出重叠，cmodel 里也是这么断言的
  if (getSrc() == getDst())
    return emitOpError("ReduceOp src/dst must not alias");

  return success();
}
