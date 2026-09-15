#include "eclipse/Conversion/Passes.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"

#include "LinalgToEclipsePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_CONVERTLINALGTOECLIPSE
#include "Passes.h.inc"

namespace {

class ConvertLinalgToEclipse
    : public impl::ConvertLinalgToEclipseBase<ConvertLinalgToEclipse> {
public:
  using impl::ConvertLinalgToEclipseBase<
      ConvertLinalgToEclipse>::ConvertLinalgToEclipseBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect,
                    EclipseDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToEclipsePatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<EclipseDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect,
                           scf::SCFDialect, tensor::TensorDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::eclipse
