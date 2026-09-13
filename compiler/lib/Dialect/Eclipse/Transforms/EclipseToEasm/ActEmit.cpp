#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

uint32_t emitActOp(Operation *op, llvm::raw_ostream &fileOS,
                   uint32_t descAddr) {
  auto actOp = mlir::cast<ActOp>(op);

  uint32_t dst = actOp.getDst().getDefiningOp<SramOp>().getAddr();
  uint32_t src = actOp.getSrc().getDefiningOp<SramOp>().getAddr();

  uint32_t n = 1;
  for (auto dim : actOp.getSrc().getType().getShape())
    n *= dim;

  uint32_t kind = static_cast<uint32_t>(actOp.getKind());

  fileOS << llvm::formatv(
      "{0,-15} desc={1:x} dst={2:x} src={3:x} n={4:d} kind={5:d}\n", "ACT",
      descAddr, dst, src, n, kind);

  return descAddr + DESC_LEN;
}

} // namespace mlir::eclipse
