#ifndef ECLIPSE_CONVERSION_PASSES
#define ECLIPSE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgToEclipse : Pass<"convert-linalg-to-eclipse", "::mlir::func::FuncOp"> {
  let summary = "Lower bufferized linalg operations to Eclipse ops";
  let description = [{
    TODO(user): 补充 pass 说明。

    目标输入形态（bufferize-function-boundaries 之后）：
      func.func @matmul(%A: memref<..., 1>, %B: memref<..., 1>, %C: memref<..., 1>) {
        linalg.matmul ins(%A, %B) outs(%C)
        return
      }

    输出目标：
      eclipse.dma_load / eclipse.matmul / eclipse.sync / eclipse.dma_store
  }];
}

#endif // ECLIPSE_CONVERSION_PASSES
