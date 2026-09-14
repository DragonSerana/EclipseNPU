#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir::eclipse {

namespace {

/// 把 src/dst 的 DDR 视图各搬一块进 SRAM，跑一条带 kind 的 ACT，再搬回。
LogicalResult lowerAct(Operation *op, PatternRewriter &rewriter, Value src,
                       Value dst, ActKind kind) {
  Location loc = op->getLoc();

  Value srcDDR = toDDR(rewriter, loc, src);
  Value dstDDR = toDDR(rewriter, loc, dst);

  auto dstType = mlir::cast<MemRefType>(dst.getType());
  auto tileType = MemRefType::get(dstType.getShape(), dstType.getElementType());

  Value srcSram = memref::AllocOp::create(rewriter, loc, tileType);
  Value dstSram = memref::AllocOp::create(rewriter, loc, tileType);

  DmaLoadOp::create(rewriter, loc, srcDDR, srcSram);
  SyncOp::create(rewriter, loc);

  auto kindAttr = ActKindAttr::get(rewriter.getContext(), kind);
  ActOp::create(rewriter, loc, srcSram, dstSram, kindAttr);

  SyncOp::create(rewriter, loc);
  DmaStoreOp::create(rewriter, loc, dstSram, dstDDR);

  memref::DeallocOp::create(rewriter, loc, dstSram);
  memref::DeallocOp::create(rewriter, loc, srcSram);

  rewriter.eraseOp(op);
  return success();
}

/// 具名一元 op（linalg.exp / linalg.rsqrt）→ eclipse.act{kind}。
template <typename LinalgOpTy>
class NamedActLowering : public OpRewritePattern<LinalgOpTy> {
public:
  NamedActLowering(MLIRContext *context, ActKind kind)
      : OpRewritePattern<LinalgOpTy>(context), kind_(kind) {}

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    return lowerAct(op, rewriter, op.getInputs()[0], op.getOutputs()[0], kind_);
  }

private:
  ActKind kind_;
};

// TODO(user): 目前把任意“单输入单输出”的 linalg.generic 当作 ReLU 处理。
// 更严格的匹配（body 确实是 arith.maximumf(x, 0)）后面再加；认不出来的必须
// 返回 failure，否则 exp 之类会被静默算成 relu。
class ActLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    return lowerAct(op, rewriter, op.getInputs()[0], op.getOutputs()[0],
                    ActKind::RELU);
  }
};

} // namespace

void populateActLowering(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<NamedActLowering<linalg::ExpOp>>(context, ActKind::EXP);
  patterns.add<NamedActLowering<linalg::RsqrtOp>>(context, ActKind::RSQRT);
  patterns.add<ActLowering>(context);
}

} // namespace mlir::eclipse
