#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

uint32_t emitMatmulOp(Operation *op, llvm::raw_ostream &fileOS,
                      uint32_t descAddr) {
  auto matmulOp = mlir::cast<MatmulOp>(op);

  uint32_t dst = matmulOp.getDst().getDefiningOp<SramOp>().getAddr();
  uint32_t lhs = matmulOp.getLhs().getDefiningOp<SramOp>().getAddr();
  uint32_t rhs = matmulOp.getRhs().getDefiningOp<SramOp>().getAddr();

  // TODO Matmul参数暂时写死
  uint32_t M = matmulOp.getLhs().getType().getShape()[0];
  uint32_t K = matmulOp.getLhs().getType().getShape()[1];
  uint32_t N = matmulOp.getRhs().getType().getShape()[1];
  uint32_t acc = matmulOp.getAccumulate();
  fileOS << llvm::formatv("{0,-14} desc={1:x} dst={2:x} lhs={3:x} rhs={4:x} "
                          "M={5:d} N={6:d} K={7:d} acc={8:d}\n",
                          "MATMUL", descAddr, dst, lhs, rhs, M, N, K, acc);

  return descAddr + DESC_LEN;
}

} // namespace mlir::eclipse
