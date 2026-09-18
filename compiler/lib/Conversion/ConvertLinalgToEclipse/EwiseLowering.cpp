#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;

namespace mlir::eclipse {

namespace {

MemRefType sramTypeOf(Value v) {
  auto t = mlir::cast<MemRefType>(v.getType());
  return MemRefType::get(t.getShape(), t.getElementType());
}

void emitEwiseSequence(PatternRewriter &rewriter, Location loc, Value big,
                       Value small, Value out,
                       function_ref<void(Value, Value, Value)> build) {
  Value bigDDR = toDDR(rewriter, loc, big);
  Value smallDDR = toDDR(rewriter, loc, small);
  Value outDDR = toDDR(rewriter, loc, out);

  Value bigSram = memref::AllocOp::create(rewriter, loc, sramTypeOf(big));
  Value smallSram = memref::AllocOp::create(rewriter, loc, sramTypeOf(small));
  Value outSram = memref::AllocOp::create(rewriter, loc, sramTypeOf(out));

  DmaLoadOp::create(rewriter, loc, bigDDR, bigSram);
  DmaLoadOp::create(rewriter, loc, smallDDR, smallSram);
  SyncOp::create(rewriter, loc);

  build(bigSram, smallSram, outSram);

  SyncOp::create(rewriter, loc);
  DmaStoreOp::create(rewriter, loc, outSram, outDDR);

  memref::DeallocOp::create(rewriter, loc, outSram);
  memref::DeallocOp::create(rewriter, loc, smallSram);
  memref::DeallocOp::create(rewriter, loc, bigSram);
}

/// linalg 的具名二元逐元素算子：形状必然相同，直接当无广播处理。
template <typename LinalgOpTy, typename EclipseOpTy>
class EwiseLowering : public OpRewritePattern<LinalgOpTy> {
public:
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    emitEwiseSequence(rewriter, loc, op.getInputs()[0], op.getInputs()[1],
                      op.getOutputs()[0], [&](Value lhs, Value rhs, Value dst) {
                        EclipseOpTy::create(rewriter, loc, lhs, rhs, dst);
                      });
    rewriter.eraseOp(op);
    return success();
  }
};

bool isIdentityMap(AffineMap m) {
  if (m.getNumDims() != 2 || m.getNumSymbols() != 0 || m.getNumResults() != 2)
    return false;
  return m.getResult(0) == getAffineDimExpr(0, m.getContext()) &&
         m.getResult(1) == getAffineDimExpr(1, m.getContext());
}

bool isConstZero(AffineExpr e) {
  auto cst = llvm::dyn_cast<AffineConstantExpr>(e);
  return cst && cst.getValue() == 0;
}

std::optional<int64_t> matchModD1(AffineExpr e) {
  auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin || bin.getKind() != AffineExprKind::Mod)
    return std::nullopt;
  if (bin.getLHS() != getAffineDimExpr(1, e.getContext()))
    return std::nullopt;
  auto cst = llvm::dyn_cast<AffineConstantExpr>(bin.getRHS());
  if (!cst)
    return std::nullopt;
  return cst.getValue();
}

bool rhsMapMatches(AffineMap m, ArrayRef<int64_t> rhsShape,
                   ArrayRef<int64_t> dstShape) {
  if (m.getNumDims() != 2 || m.getNumSymbols() != 0 || m.getNumResults() != 2)
    return false;

  bool rowOk = (rhsShape[0] == 1)
                   ? isConstZero(m.getResult(0))
                   : (m.getResult(0) == getAffineDimExpr(0, m.getContext()));
  if (!rowOk)
    return false;

  if (rhsShape[1] == 1)
    return isConstZero(m.getResult(1));
  if (rhsShape[1] == dstShape[1])
    return m.getResult(1) == getAffineDimExpr(1, m.getContext()) ||
           matchModD1(m.getResult(1)) == rhsShape[1];
  return matchModD1(m.getResult(1)) == rhsShape[1];
}

enum class BinOp { Add, Sub, Mul, Div };

std::optional<BinOp> binOpOf(Operation *op) {
  if (isa<arith::AddFOp>(op))
    return BinOp::Add;
  if (isa<arith::SubFOp>(op))
    return BinOp::Sub;
  if (isa<arith::MulFOp>(op))
    return BinOp::Mul;
  if (isa<arith::DivFOp>(op))
    return BinOp::Div;
  return std::nullopt;
}

class EwiseBroadcastLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();
    if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        }))
      return failure();

    Block &body = op.getRegion().front();
    auto yieldOp = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1)
      return failure();
    Operation *compute = yieldOp.getValues()[0].getDefiningOp();
    if (!compute || compute->getBlock() != &body ||
        compute->getNumOperands() != 2)
      return failure();
    auto bin = binOpOf(compute);
    if (!bin)
      return failure();

    auto argIndex = [&](Value v) -> int {
      auto arg = llvm::dyn_cast<BlockArgument>(v);
      if (!arg || arg.getOwner() != &body)
        return -1;
      return static_cast<int>(arg.getArgNumber());
    };
    int aIdx = argIndex(compute->getOperand(0));
    int bIdx = argIndex(compute->getOperand(1));
    if (aIdx < 0 || bIdx < 0)
      return failure();

    auto in0Type = mlir::dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto in1Type = mlir::dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto outType = mlir::dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!in0Type || !in1Type || !outType || !in0Type.hasStaticShape() ||
        !in1Type.hasStaticShape() || !outType.hasStaticShape())
      return failure();
    if (outType.getRank() != 2)
      return failure();

    // 主操作数是和输出同形的那个；另一个是小操作数（描述符里的 rhs）。
    int bigIdx, smallIdx;
    if (in0Type.getShape() == outType.getShape()) {
      bigIdx = 0;
      smallIdx = 1;
    } else if (in1Type.getShape() == outType.getShape()) {
      bigIdx = 1;
      smallIdx = 0;
    } else {
      return failure();
    }

    // body 的操作数必须恰好是 {big, small}，一个不多一个不少。少了这条，
    // out+out、rhs+rhs 这类 body 会被静默降成 dst = lhs op rhs。
    bool forward = (aIdx == bigIdx && bIdx == smallIdx);
    bool swapped = (aIdx == smallIdx && bIdx == bigIdx);
    if (!forward && !swapped)
      return failure();
    if (swapped && *bin != BinOp::Add && *bin != BinOp::Mul)
      return failure();

    auto smallType = mlir::cast<MemRefType>(op.getInputs()[smallIdx].getType());
    if (smallType.getRank() != 2 || smallType.getShape()[1] <= 0 ||
        outType.getShape()[1] % smallType.getShape()[1] != 0)
      return failure();
    if (smallType.getShape()[0] != 1 &&
        smallType.getShape()[0] != outType.getShape()[0])
      return failure();

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 || !isIdentityMap(maps[bigIdx]) ||
        !isIdentityMap(maps[2]) ||
        !rhsMapMatches(maps[smallIdx], smallType.getShape(),
                       outType.getShape()))
      return failure();

    Location loc = op.getLoc();
    BinOp kind = *bin;
    emitEwiseSequence(rewriter, loc, op.getInputs()[bigIdx],
                      op.getInputs()[smallIdx], op.getOutputs()[0],
                      [&](Value lhs, Value rhs, Value dst) {
                        switch (kind) {
                        case BinOp::Add:
                          EwiseAddOp::create(rewriter, loc, lhs, rhs, dst);
                          break;
                        case BinOp::Sub:
                          EwiseSubOp::create(rewriter, loc, lhs, rhs, dst);
                          break;
                        case BinOp::Mul:
                          EwiseMulOp::create(rewriter, loc, lhs, rhs, dst);
                          break;
                        case BinOp::Div:
                          EwiseDivOp::create(rewriter, loc, lhs, rhs, dst);
                          break;
                        }
                      });
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateEwiseLowering(RewritePatternSet &patterns) {
  patterns
      .add<EwiseLowering<linalg::AddOp, EwiseAddOp>,
           EwiseLowering<linalg::SubOp, EwiseSubOp>,
           EwiseLowering<linalg::MulOp, EwiseMulOp>,
           EwiseLowering<linalg::DivOp, EwiseDivOp>, EwiseBroadcastLowering>(
          patterns.getContext());
}

} // namespace mlir::eclipse
