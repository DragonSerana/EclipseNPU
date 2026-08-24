#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value dst = op.getOutputs()[0];

    Value lhsDDR = toDDR(rewriter, loc, lhs);
    Value rhsDDR = toDDR(rewriter, loc, rhs);
    Value dstDDR = toDDR(rewriter, loc, dst);

    auto lhsType = mlir::cast<MemRefType>(lhs.getType());
    auto rhsType = mlir::cast<MemRefType>(rhs.getType());
    auto dstType = mlir::cast<MemRefType>(dst.getType());

    auto sramLhsType =
        MemRefType::get(lhsType.getShape(), lhsType.getElementType());
    auto sramRhsType =
        MemRefType::get(rhsType.getShape(), rhsType.getElementType());
    auto sramDstType =
        MemRefType::get(dstType.getShape(), dstType.getElementType());

    Value lhsSram = memref::AllocOp::create(rewriter, loc, sramLhsType);
    Value rhsSram = memref::AllocOp::create(rewriter, loc, sramRhsType);
    Value dstSram = memref::AllocOp::create(rewriter, loc, sramDstType);

    DmaLoadOp::create(rewriter, loc, lhsDDR, lhsSram);
    DmaLoadOp::create(rewriter, loc, rhsDDR, rhsSram);
    SyncOp::create(rewriter, loc);

    // TODO.暂时写死False
    auto accumulate = rewriter.getBoolAttr(false);
    MatmulOp::create(rewriter, loc, lhsSram, rhsSram, dstSram, accumulate);

    SyncOp::create(rewriter, loc);
    DmaStoreOp::create(rewriter, loc, dstSram, dstDDR);

    memref::DeallocOp::create(rewriter, loc, dstSram);
    memref::DeallocOp::create(rewriter, loc, rhsSram);
    memref::DeallocOp::create(rewriter, loc, lhsSram);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMatmulLowering(RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
