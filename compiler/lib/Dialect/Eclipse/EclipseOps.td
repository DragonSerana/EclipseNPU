#ifndef ECLIPSE_OPS
#define ECLIPSE_OPS

include "mlir/IR/OpBase.td"

// 定义 Dialect 的名字
def Eclipse_Dialect : Dialect {
  let name = "eclipse";
  let summary = "Eclipse NPU Dialect";
}

#endif // ECLIPSE_OPS
