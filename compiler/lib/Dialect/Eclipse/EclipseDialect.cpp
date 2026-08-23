#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::eclipse::EclipseDialect)

using namespace mlir;
using namespace mlir::eclipse;

#include "EclipseOpsEnums.cpp.inc"

#define GET_OP_CLASSES
#include "EclipseOps.cpp.inc"

EclipseDialect::EclipseDialect(MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<EclipseDialect>()) {
  initialize();
}

void EclipseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "EclipseOps.cpp.inc"
      >();
}
