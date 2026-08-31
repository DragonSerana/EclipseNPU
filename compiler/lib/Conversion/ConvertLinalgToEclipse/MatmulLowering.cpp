#include "LinalgToEclipseCommon.h"
#include "LinalgToEclipsePatterns.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

namespace {

constexpr uint32_t tileK = 16;

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

    auto m = lhsType.getShape()[0];
    auto k = lhsType.getShape()[1];
    auto n = rhsType.getShape()[1];

    llvm::SmallVector<int64_t, 4> lhsShape = {m, tileK};        
    llvm::SmallVector<int64_t, 4> rhsShape = {tileK, n};   
    llvm::SmallVector<int64_t, 4> resShape = {m, n};                           
    
    auto sramLhsTileType =
        MemRefType::get(lhsShape, lhsType.getElementType());
    auto sramRhsTileType =
        MemRefType::get(rhsShape, rhsType.getElementType());
    auto sramDstTileType =
        MemRefType::get(resShape, dstType.getElementType());    

    if (lhsType.getShape()[1]%tileK == 0) {
      auto tile = lhsType.getShape()[1]/tileK;
      Value lhsSram = memref::AllocOp::create(rewriter, loc, sramLhsTileType);
      Value rhsSram = memref::AllocOp::create(rewriter, loc, sramRhsTileType);
      Value dstSram = memref::AllocOp::create(rewriter, loc, sramDstTileType);

      // 剥离首块，不放入scf，不累加，避免SRAM中的垃圾值
      SmallVector<OpFoldResult> offsets0Lhs = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};     
      SmallVector<OpFoldResult> sizes0Lhs   = {rewriter.getIndexAttr(m), rewriter.getIndexAttr(tileK)};
      SmallVector<OpFoldResult> strides0Lhs = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

      Value sub0Lhs = rewriter.create<memref::SubViewOp>(loc, lhsDDR, offsets0Lhs, sizes0Lhs, strides0Lhs);

      SmallVector<OpFoldResult> offsets0Rhs = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};     
      SmallVector<OpFoldResult> sizes0Rhs   = {rewriter.getIndexAttr(tileK), rewriter.getIndexAttr(n)};
      SmallVector<OpFoldResult> strides0Rhs = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

      Value sub0Rhs = rewriter.create<memref::SubViewOp>(loc, rhsDDR, offsets0Rhs, sizes0Rhs, strides0Rhs);        

      DmaLoadOp::create(rewriter, loc, sub0Lhs, lhsSram);
      DmaLoadOp::create(rewriter, loc, sub0Rhs, rhsSram);
      SyncOp::create(rewriter, loc);

      MatmulOp::create(rewriter, loc, lhsSram, rhsSram, dstSram, rewriter.getBoolAttr(false));
      SyncOp::create(rewriter, loc);

      for (uint32_t i = 0; i < tile; i++) {
        SmallVector<OpFoldResult> offsetsLhs = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(i*tileK)};     
        SmallVector<OpFoldResult> sizesLhs   = {rewriter.getIndexAttr(m), rewriter.getIndexAttr(tileK)};
        SmallVector<OpFoldResult> stridesLhs = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value subLhs = rewriter.create<memref::SubViewOp>(loc, lhsDDR, offsetsLhs, sizesLhs, stridesLhs);

        SmallVector<OpFoldResult> offsetsRhs = {rewriter.getIndexAttr(i*tileK), rewriter.getIndexAttr(0)};     
        SmallVector<OpFoldResult> sizesRhs   = {rewriter.getIndexAttr(tileK), rewriter.getIndexAttr(n)};
        SmallVector<OpFoldResult> stridesRhs = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

        Value subRhs = rewriter.create<memref::SubViewOp>(loc, rhsDDR, offsetsRhs, sizesRhs, stridesRhs);        

        DmaLoadOp::create(rewriter, loc, subLhs, lhsSram);
        DmaLoadOp::create(rewriter, loc, subRhs, rhsSram);
        SyncOp::create(rewriter, loc);

        BoolAttr accumulate;
        if (i == 0) 
          accumulate = rewriter.getBoolAttr(false);
        else 
          accumulate = rewriter.getBoolAttr(true);

        MatmulOp::create(rewriter, loc, lhsSram, rhsSram, dstSram, accumulate);

        SyncOp::create(rewriter, loc);
      }
      
      DmaStoreOp::create(rewriter, loc, dstSram, dstDDR);

      memref::DeallocOp::create(rewriter, loc, dstSram);
      memref::DeallocOp::create(rewriter, loc, rhsSram);
      memref::DeallocOp::create(rewriter, loc, lhsSram);
    } else {
      llvm_unreachable("目前只处理可以整除的场景");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMatmulLowering(RewritePatternSet &patterns) {
  patterns.add<MatmulLowering>(patterns.getContext());
}

} // namespace mlir::eclipse
