#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

uint32_t emitReduceOp(Operation *op, llvm::raw_ostream &fileOS,
                      uint32_t descAddr) {
  auto reduceOp = mlir::cast<ReduceOp>(op);

  uint32_t dst = reduceOp.getDst().getDefiningOp<SramOp>().getAddr();
  uint32_t src = reduceOp.getSrc().getDefiningOp<SramOp>().getAddr();
  auto srcShape =
      mlir::cast<MemRefType>(reduceOp.getSrc().getType()).getShape();
  uint32_t kind = static_cast<uint32_t>(reduceOp.getKind());

  if (Value idx = reduceOp.getIdx()) {
    uint32_t idxAddr = idx.getDefiningOp<SramOp>().getAddr();
    fileOS << llvm::formatv("{0,-15} desc={1:x} dst={2:x} idx={3:x} src={4:x} "
                            "rows={5:d} cols={6:d} kind={7:d}\n",
                            "REDUCE", descAddr, dst, idxAddr, src, srcShape[0],
                            srcShape[1], kind);
  } else {
    fileOS << llvm::formatv("{0,-15} desc={1:x} dst={2:x} src={3:x} rows={4:d} "
                            "cols={5:d} kind={6:d}\n",
                            "REDUCE", descAddr, dst, src, srcShape[0],
                            srcShape[1], kind);
  }

  return descAddr + DESC_LEN;
}

} // namespace mlir::eclipse
