#ifndef ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H
#define ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace eclipse {

/// 注册所有 linalg -> Eclipse 的 lowering pattern。
void populateLinalgToEclipsePatterns(RewritePatternSet &patterns);

/// 单个算子的 pattern 注册入口，后续每加一个算子就加一个。
void populateMatmulLowering(RewritePatternSet &patterns);
void populateAddLowering(RewritePatternSet &patterns);

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H
