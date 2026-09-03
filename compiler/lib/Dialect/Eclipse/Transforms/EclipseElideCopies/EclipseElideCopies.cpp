#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "eclipse/Dialect/Eclipse/Transforms/ElideCopiesPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSEELIDECOPIES
#include "EclipseElideCopiesPasses.h.inc"

namespace {

class EclipseElideCopies
    : public impl::EclipseElideCopiesBase<EclipseElideCopies> {
public:
  using impl::EclipseElideCopiesBase<EclipseElideCopies>::EclipseElideCopiesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

  }
};

} // namespace

} // namespace mlir::eclipse
