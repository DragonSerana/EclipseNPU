#ifndef ECLIPSE_DIALECT_ECLIPSEDIALECT_H
#define ECLIPSE_DIALECT_ECLIPSEDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace eclipse {

class EclipseDialect : public ::mlir::Dialect {
public:
  explicit EclipseDialect(::mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "eclipse"; }

private:
  void initialize();
};

} // namespace eclipse
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::eclipse::EclipseDialect)

#endif // ECLIPSE_DIALECT_ECLIPSEDIALECT_H
