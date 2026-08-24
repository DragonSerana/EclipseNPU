#include "LinalgToEclipseCommon.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir::eclipse {

Value toDDR(PatternRewriter &rewriter, Location loc, Value value) {
  auto type = mlir::cast<MemRefType>(value.getType());
  // 设置memref类型为ddr
  auto ddrType =
      MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
                      rewriter.getI64IntegerAttr(1));
  return memref::MemorySpaceCastOp::create(rewriter, loc, ddrType, value);
}

} // namespace mlir::eclipse
