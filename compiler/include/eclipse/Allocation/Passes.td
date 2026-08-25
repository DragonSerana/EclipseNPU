#ifndef ECLIPSE_ALLOCATION_PASSES
#define ECLIPSE_ALLOCATION_PASSES

include "mlir/Pass/PassBase.td"

def EclipseAllocate : Pass<"eclipse-allocate", "::mlir::ModuleOp"> {
  let summary = "Allocate SRAM buffers and bind DDR ABI addresses";
  let description = [{
    TODO(user): 补充 eclipse-allocate 的说明。

    这个 pass 负责：
    - memref.alloc(space 0) -> eclipse.sram {addr}
    - 给函数参数挂 DDR ABI 地址
    - 地址级检查（对齐 / 驻留 / 重叠）
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
