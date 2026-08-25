#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

// TODO(user): 目前把任意“单输入单输出”的 linalg.generic 当作 ReLU 处理。
// 更严格的匹配（body 确实是 relu）后面再加。
class ActLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    Value src = op.getInputs()[0];
    Value dst = op.getOutputs()[0];

    Value srcDDR = toDDR(rewriter, loc, src);
    Value dstDDR = toDDR(rewriter, loc, dst);

    auto dstType = mlir::cast<MemRefType>(dst.getType());
    auto sramSrcType =
        MemRefType::get(dstType.getShape(), dstType.getElementType());
    auto sramDstType =
        MemRefType::get(dstType.getShape(), dstType.getElementType());

    Value srcSram = memref::AllocOp::create(rewriter, loc, sramSrcType);
    Value dstSram = memref::AllocOp::create(rewriter, loc, sramDstType);

    DmaLoadOp::create(rewriter, loc, srcDDR, srcSram);
    SyncOp::create(rewriter, loc);

    // TODO.暂时写死RELU
    auto kind = ActKindAttr::get(rewriter.getContext(), ActKind::RELU);
    ActOp::create(rewriter, loc, srcSram, dstSram, kind);

    SyncOp::create(rewriter, loc);
    DmaStoreOp::create(rewriter, loc, dstSram, dstDDR);

    memref::DeallocOp::create(rewriter, loc, dstSram);
    memref::DeallocOp::create(rewriter, loc, srcSram);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateActLowering(RewritePatternSet &patterns) {
  patterns.add<ActLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
