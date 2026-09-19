#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include <optional>

using namespace mlir;

namespace mlir::eclipse {

namespace {

/// 把 src/dst 的 DDR 视图各搬一块进 SRAM，跑一条带 kind 的 REDUCE，再搬回。
LogicalResult lowerReduce(Operation *op, PatternRewriter &rewriter, Value src,
                          Value dst, ReduceKind kind) {
  Location loc = op->getLoc();

  Value srcDDR = toDDR(rewriter, loc, src);
  Value dstDDR = toDDR(rewriter, loc, dst);

  auto srcType = mlir::cast<MemRefType>(src.getType());
  auto dstType = mlir::cast<MemRefType>(dst.getType());
  auto srcTileType =
      MemRefType::get(srcType.getShape(), srcType.getElementType());
  auto dstTileType =
      MemRefType::get(dstType.getShape(), dstType.getElementType());

  Value srcSram = memref::AllocOp::create(rewriter, loc, srcTileType);
  Value dstSram = memref::AllocOp::create(rewriter, loc, dstTileType);

  DmaLoadOp::create(rewriter, loc, srcDDR, srcSram);
  SyncOp::create(rewriter, loc);

  auto kindAttr = ReduceKindAttr::get(rewriter.getContext(), kind);
  ReduceOp::create(rewriter, loc, srcSram, dstSram, Value(), kindAttr);

  SyncOp::create(rewriter, loc);
  DmaStoreOp::create(rewriter, loc, dstSram, dstDDR);

  memref::DeallocOp::create(rewriter, loc, dstSram);
  memref::DeallocOp::create(rewriter, loc, srcSram);

  rewriter.eraseOp(op);
  return success();
}

/// 从 combiner body 认 kind。in 是输入元素，out 是累加器。
std::optional<ReduceKind> matchReduceBody(Block *body, Value in, Value out) {
  Operation *root = body->getTerminator()->getOperand(0).getDefiningOp();
  if (!root)
    return std::nullopt;

  if (auto add = dyn_cast<arith::AddFOp>(root)) {
    // out + in        -> SUM
    // out + (in * in) -> SQUARE_SUM（平方折进归约输入端）
    if (add.getLhs() != out && add.getRhs() != out)
      return std::nullopt;
    Value other = (add.getLhs() == out) ? add.getRhs() : add.getLhs();
    if (other == in)
      return ReduceKind::SUM;
    if (auto mul = other.getDefiningOp<arith::MulFOp>())
      if (mul.getLhs() == in && mul.getRhs() == in)
        return ReduceKind::SQUARE_SUM;
    return std::nullopt;
  }

  if (isa<arith::MaximumFOp, arith::MaxNumFOp>(root)) {
    const bool pair =
        (root->getOperand(0) == in && root->getOperand(1) == out) ||
        (root->getOperand(0) == out && root->getOperand(1) == in);
    if (pair)
      return ReduceKind::MAX;
  }

  return std::nullopt;
}

/// linalg.generic 写的 rowwise 归约：
///   indexing_maps = [(d0,d1)->(d0,d1), (d0,d1)->(d0,0)]
///   iterator_types = ["parallel", "reduction"]
/// 输出就是 [rows, 1]，和 ISA 的 REDUCE 一一对应。
/// init 只用来定输出缓冲，ISA 的 REDUCE 从第一个元素起归约，不看 init 的内容。
class GenericReduceLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    auto iterators = op.getIteratorTypesArray();
    if (iterators.size() != 2 ||
        iterators[0] != utils::IteratorType::parallel ||
        iterators[1] != utils::IteratorType::reduction)
      return failure();

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || maps[0].getNumResults() != 2 ||
        maps[1].getNumResults() != 2)
      return failure();
    if (maps[0] != AffineMap::getMultiDimIdentityMap(2, op.getContext()))
      return failure();
    // 输出必须是 [rows, 1]：第二个下标恒为常量 0
    auto zero = dyn_cast<AffineConstantExpr>(maps[1].getResult(1));
    if (!zero || zero.getValue() != 0)
      return failure();

    Block *body = op.getBody();
    std::optional<ReduceKind> kind =
        matchReduceBody(body, body->getArgument(0), body->getArgument(1));
    if (!kind)
      return failure();

    return lowerReduce(op, rewriter, op.getInputs()[0], op.getOutputs()[0],
                       *kind);
  }
};

} // namespace

void populateReduceLowering(RewritePatternSet &patterns) {
  patterns.add<GenericReduceLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
