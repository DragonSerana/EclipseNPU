#include "eclipse/Dialect/Eclipse/EclipseDialect.h"
#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "llvm/Support/Format.h" 

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::eclipse::EclipseDialect)

using namespace mlir;
using namespace mlir::eclipse;

#include "EclipseOpsEnums.cpp.inc"

static void printPrintHex(OpAsmPrinter &p, SramOp op, IntegerAttr addr) {
  p << llvm::format("0x%x", addr.getValue().getZExtValue());
}

static mlir::ParseResult parsePrintHex(mlir::OpAsmParser &parser,
                                       mlir::IntegerAttr &addr) {
  int64_t value;
  if (parser.parseInteger(value))
    return mlir::failure();
  auto builder = parser.getBuilder();
  addr = builder.getI32IntegerAttr(static_cast<int32_t>(value));
  return mlir::success();
}

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
