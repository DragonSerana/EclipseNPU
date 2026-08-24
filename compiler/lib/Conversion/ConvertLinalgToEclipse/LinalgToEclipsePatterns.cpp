#include "LinalgToEclipsePatterns.h"

namespace mlir::eclipse {

void populateLinalgToEclipsePatterns(RewritePatternSet &patterns) {
  populateMatmulLowering(patterns);
  populateAddLowering(patterns);
}

} // namespace mlir::eclipse
