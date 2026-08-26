#ifndef ECLIPSE_EMIT_PASSES
#define ECLIPSE_EMIT_PASSES

include "mlir/Pass/PassBase.td"

def EclipseToEasm : Pass<"eclipse-to-easm", "::mlir::ModuleOp"> {
  let summary = "Emit Eclipse IR to .easm text";
  let description = [{
    TODO: 补充说明。
  }];

  let options = [
    Option<"output", "o", "std::string", /*default=*/"\"\"",
           "Output .easm path">
  ];

  let dependentDialects = [
    "mlir::eclipse::EclipseDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // ECLIPSE_EMIT_PASSES
