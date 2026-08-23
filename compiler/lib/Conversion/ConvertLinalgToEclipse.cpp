#include "eclipse/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

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
    // TODO(user): 在这里实现 linalg.matmul -> Eclipse 的转换逻辑。
    // 当前先留空，确保 pass 能注册、能挂在 eclipse-opt 上。
  }
};

} // namespace

} // namespace mlir::eclipse
