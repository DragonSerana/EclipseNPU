#include "LinalgToEclipsePatterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(user): 在这里把 linalg.matmul lower 成
    // dma_load + sync + eclipse.matmul + sync + dma_store。
    return failure();
  }
};

} // namespace

void populateMatmulLowering(RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
