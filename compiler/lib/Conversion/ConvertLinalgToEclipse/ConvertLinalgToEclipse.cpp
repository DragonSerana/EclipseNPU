#include "eclipse/Conversion/Passes.h"
#include "eclipse/Dialect/Eclipse/EclipseDialect.h"

#include "LinalgToEclipsePatterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_CONVERTLINALGTOECLIPSE
#include "Passes.h.inc"

namespace {

class ConvertLinalgToEclipse
    : public impl::ConvertLinalgToEclipseBase<ConvertLinalgToEclipse> {
public:
  using impl::ConvertLinalgToEclipseBase<
      ConvertLinalgToEclipse>::ConvertLinalgToEclipseBase;

  void runOnOperation() override {
    getContext().getOrLoadDialect<EclipseDialect>();

    RewritePatternSet patterns(&getContext());
    populateLinalgToEclipsePatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::eclipse
