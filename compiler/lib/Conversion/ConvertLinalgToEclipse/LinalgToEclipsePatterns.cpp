#include "LinalgToEclipsePatterns.h"

namespace mlir::eclipse {

void populateLinalgToEclipsePatterns(RewritePatternSet &patterns) {
  populateMatmulLowering(patterns);
  populateEwiseLowering(patterns);
  populateActLowering(patterns);
  populateReduceLowering(patterns);
}

} // namespace mlir::eclipse
