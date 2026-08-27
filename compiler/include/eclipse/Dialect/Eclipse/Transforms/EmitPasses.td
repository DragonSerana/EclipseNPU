#ifndef ECLIPSE_EMIT_PASSES
#define ECLIPSE_EMIT_PASSES

include "mlir/Pass/PassBase.td"

def EclipseToEasm : Pass<"eclipse-to-easm", "::mlir::ModuleOp"> {
  let summary = "Emit Eclipse IR to .easm text";
  let description = [{
    Pure serializer: emits eclipse ops as .easm text in program order.
    No transformation; expects upstream passes to have allocated SRAM
    and bound DDR addresses.
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
