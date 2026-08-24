#include "LinalgToEclipsePatterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

class AddLowering : public OpRewritePattern<linalg::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(user): 如果后续要支持 linalg.add -> eclipse.elementwise_add，
    // 在这里实现。
    return failure();
  }
};

} // namespace

void populateAddLowering(RewritePatternSet &patterns) {
  patterns.add<AddLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
