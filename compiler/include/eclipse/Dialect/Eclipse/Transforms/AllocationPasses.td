#ifndef ECLIPSE_ALLOCATION_PASSES
#define ECLIPSE_ALLOCATION_PASSES

include "mlir/Pass/PassBase.td"

def EclipseAllocate : Pass<"eclipse-allocate", "::mlir::ModuleOp"> {
  let summary = "Allocate SRAM buffers and bind DDR ABI addresses";
  let description = [{
    Lowers memref.alloc(space 0) to eclipse.sram with assigned addresses,
    attaches DDR ABI addresses to function arguments, and checks
    alignment / residency / overlap.
  }];

  let options = [
    Option<"layout", "layout", "std::string", /*default=*/"\"bump\"",
           "SRAM layout: bump or golden-mirror">
  ];

  let dependentDialects = [
    "mlir::eclipse::EclipseDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // ECLIPSE_ALLOCATION_PASSES
