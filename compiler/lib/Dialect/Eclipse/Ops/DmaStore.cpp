#include "eclipse/Dialect/Eclipse/EclipseOps.h"

using namespace mlir;
using namespace mlir::eclipse;

LogicalResult DmaStoreOp::verify() {
  // TODO: 填写 dma_store 的结构检查。
  // 例如：
  // - src 是 memory_space 0 (SRAM)
  // - dst 是 memory_space 1 (DDR)
  // - 元素类型是 f16
  // - src 是 packed
  // - 形状为二维且 src/dst 形状一致
  return success();
}
