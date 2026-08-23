#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

static bool isPacked(MemRefType type) {
  // 如果是行主序，肯定就是packed
  if (type.getLayout().isIdentity())
    return true;

  auto [strides, offset] = type.getStridesAndOffset();
  ArrayRef<int64_t> shape = type.getShape();

  if(strides.back() != 1)
    return false;

  for(int i = shape.size() - 1; i > 0; i--) {
    if (strides[i-1] != strides[i] * shape[i]){
      return false;
    }
  }

  return true;
}

LogicalResult DmaLoadOp::verify() {
  auto srcType = mlir::dyn_cast<MemRefType>(getSrc().getType());
  auto dstType = mlir::dyn_cast<MemRefType>(getDst().getType());

  if (!srcType || !dstType)
    return emitOpError("operands must be MemRef types");

  // 检查元素类型和shape是否一致
  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("source and destination element types must match (")
          << srcType.getElementType() << " vs " << dstType.getElementType()
          << ")";
  }

  if (srcType.getShape() != dstType.getShape()) {
    return emitOpError("source and destination shapes must match (")
          << srcType.getShape() << " vs " << dstType.getShape()
          << ")";
  }

  // src (DDR) must be memory space 1
  auto srcMemSpace = srcType.getMemorySpace();
  if (!srcMemSpace || !llvm::isa<IntegerAttr>(srcMemSpace))
    return emitOpError("src memory space must be an integer");
  if (llvm::cast<IntegerAttr>(srcMemSpace).getInt() != 1)
    return emitOpError("src (DDR) must be in memory space 1");

  // dst (SRAM) must be memory space 0 and identity layout (packed)
  auto dstMemSpace = dstType.getMemorySpace();
  if (!dstMemSpace || !llvm::isa<IntegerAttr>(dstMemSpace))
    return emitOpError("dst memory space must be an integer");
  if (llvm::cast<IntegerAttr>(dstMemSpace).getInt() != 0)
    return emitOpError("dst (SRAM) must be in memory space 0");



  if (!isPacked(dstType))
    return emitOpError("dst (SRAM) must have identity layout (no strides)");

  return success();
}
