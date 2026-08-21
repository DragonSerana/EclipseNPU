#ifndef ECLIPSE_OPS
#define ECLIPSE_OPS

include "mlir/IR/OpBase.td"

def Eclipse_Dialect : Dialect {
  let name = "eclipse";
  let summary = "Eclipse NPU Dialect";
  let cppNamespace = "::mlir::eclipse";
}

class Eclipse_Op<string mnemonic, list<Trait> traits = []>
    : Op<Eclipse_Dialect, mnemonic, traits>;

// 六个 leaf op 的定义将在这里逐个补充：
//   eclipse.dma_load
//   eclipse.dma_store
//   eclipse.matmul
//   eclipse.elementwise_add
//   eclipse.act
//   eclipse.sync

def AddOp : Eclipse_Op<"dma_load"> {
  let summary = "";
  let description = [{

  }];

  let arguments = (ins 
    FP16Tensor:$lhs, 
    FP16Tensor:$rhs
  );
  let results = (outs FP16Tensor);

}

#endif // ECLIPSE_OPS

