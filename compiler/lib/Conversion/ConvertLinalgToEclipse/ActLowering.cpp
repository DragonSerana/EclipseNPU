#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include <optional>

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

class ActLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    std::optional<ActKind> kind = matchActBody(op);
    if (!kind)
      return failure();

    return lowerAct(op, rewriter, op.getInputs()[0], op.getOutputs()[0], *kind);
  }

private:
  /// 目前只认 `max(x, 0)`（relu）。maximumf 和 maxnumf 都接，因为 canonicalize
  /// 或前端可能给出任一个。
  static std::optional<ActKind> matchActBody(linalg::GenericOp op) {
    Block *body = op.getBody();
    Operation *root = body->getTerminator()->getOperand(0).getDefiningOp();
    if (!root)
      return std::nullopt;

    Value input = body->getArgument(0);
    Value lhs, rhs;
    if (auto maxOp = dyn_cast<arith::MaximumFOp>(root)) {
      lhs = maxOp.getLhs();
      rhs = maxOp.getRhs();
    } else if (auto maxOp = dyn_cast<arith::MaxNumFOp>(root)) {
      lhs = maxOp.getLhs();
      rhs = maxOp.getRhs();
    } else {
      return std::nullopt;
    }

    if ((lhs == input && matchPattern(rhs, m_AnyZeroFloat())) ||
        (rhs == input && matchPattern(lhs, m_AnyZeroFloat())))
      return ActKind::RELU;
    return std::nullopt;
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
