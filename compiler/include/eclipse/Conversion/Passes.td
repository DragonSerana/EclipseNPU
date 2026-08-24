#ifndef ECLIPSE_CONVERSION_PASSES
#define ECLIPSE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgToEclipse : Pass<"convert-linalg-to-eclipse", "::mlir::func::FuncOp"> {
  let summary = "Lower bufferized linalg operations to Eclipse ops";
  let dependentDialects = [
    "mlir::eclipse::EclipseDialect",
    "mlir::memref::MemRefDialect"
  ];
  let description = [{
    Lowering linalg to eclipse dialect
  }];
}

#endif // ECLIPSE_CONVERSION_PASSES
