#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

static bool isPacked(MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;

  auto [strides, offset] = type.getStridesAndOffset();
  ArrayRef<int64_t> shape = type.getShape();

  if (strides.back() != 1)
    return false;

  for (int64_t i = shape.size() - 1; i > 0; i--) {
    if (strides[i - 1] != strides[i] * shape[i])
      return false;
  }

  return true;
}

LogicalResult DmaStoreOp::verify() {
  auto srcType = mlir::dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!srcType || !dstType)
    return emitOpError("DmaStoreOp operands must be MemRef types");

  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError(
               "DmaStoreOp source and destination element types must match (")
           << srcType.getElementType() << " vs " << dstType.getElementType()
           << ")";
  }

  if (srcType.getShape() != dstType.getShape()) {
    return emitOpError("DmaStoreOp source and destination shapes must match (")
           << srcType.getShape() << " vs " << dstType.getShape() << ")";
  }

  // src (SRAM) must be packed.
  if (!isPacked(srcType))
    return emitOpError("src (SRAM) must be packed");

  return success();
}
