#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

/// linalg 的二元逐元素算子统一降到对应的 eclipse.elementwise_*：把 lhs/rhs/dst
/// 各切一块搬进 SRAM，算完再搬回。四个算子的形状规则一样，用模板复用。
template <typename LinalgOpTy, typename EclipseOpTy>
class EwiseLowering : public OpRewritePattern<LinalgOpTy> {
public:
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value dst = op.getOutputs()[0];

    Value lhsDDR = toDDR(rewriter, loc, lhs);
    Value rhsDDR = toDDR(rewriter, loc, rhs);
    Value dstDDR = toDDR(rewriter, loc, dst);

    auto dstType = mlir::cast<MemRefType>(dst.getType());
    auto tileType =
        MemRefType::get(dstType.getShape(), dstType.getElementType());

    Value lhsSram = memref::AllocOp::create(rewriter, loc, tileType);
    Value rhsSram = memref::AllocOp::create(rewriter, loc, tileType);
    Value dstSram = memref::AllocOp::create(rewriter, loc, tileType);

    DmaLoadOp::create(rewriter, loc, lhsDDR, lhsSram);
    DmaLoadOp::create(rewriter, loc, rhsDDR, rhsSram);
    SyncOp::create(rewriter, loc);

    EclipseOpTy::create(rewriter, loc, lhsSram, rhsSram, dstSram);

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

void populateEwiseLowering(RewritePatternSet &patterns) {
  patterns.add<EwiseLowering<linalg::AddOp, EwiseAddOp>,
               EwiseLowering<linalg::SubOp, EwiseSubOp>,
               EwiseLowering<linalg::MulOp, EwiseMulOp>,
               EwiseLowering<linalg::DivOp, EwiseDivOp>>(
      patterns.getContext());
}

} // namespace mlir::eclipse
