#include "eclipse/Dialect/Eclipse/Transforms/EmitPasses.h"

#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::eclipse {

#define GEN_PASS_DEF_ECLIPSETOEASM
#include "EclipseEmitPasses.h.inc"

namespace {

class EclipseToEasm : public impl::EclipseToEasmBase<EclipseToEasm> {
public:
  using impl::EclipseToEasmBase<EclipseToEasm>::EclipseToEasmBase;

  void runOnOperation() override {
    // TODO: 将 Eclipse IR 发射为 .easm 文本。
  }
};

} // namespace

} // namespace mlir::eclipse
