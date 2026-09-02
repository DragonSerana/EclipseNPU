#ifndef ECLIPSE_CONVERTLINALGTOECLIPSE_COMMON_H
#define ECLIPSE_CONVERTLINALGTOECLIPSE_COMMON_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace eclipse {

constexpr uint32_t tileK = 16;

/// 把 bufferize 后默认空间 0 的 memref 显式转成 DDR（空间 1）。
Value toDDR(PatternRewriter &rewriter, Location loc, Value value);

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_CONVERTLINALGTOECLIPSE_COMMON_H
