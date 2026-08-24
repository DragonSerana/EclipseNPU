#ifndef ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H
#define ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace eclipse {

void populateLinalgToEclipsePatterns(RewritePatternSet &patterns);

void populateMatmulLowering(RewritePatternSet &patterns);
void populateAddLowering(RewritePatternSet &patterns);
void populateActLowering(RewritePatternSet &patterns);

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_CONVERTLINALGTOECLIPSE_PATTERNS_H
