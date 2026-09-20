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

  // 转置时对应矩阵在描述符里按 [K,M] / [N,K] 摆布，逻辑上仍算 MxN
  auto lhsShape = matmulOp.getLhs().getType().getShape();
  auto rhsShape = matmulOp.getRhs().getType().getShape();
  const bool transA = matmulOp.getTransA();
  const bool transB = matmulOp.getTransB();

  uint32_t M = transA ? lhsShape[1] : lhsShape[0];
  uint32_t K = transA ? lhsShape[0] : lhsShape[1];
  uint32_t N = transB ? rhsShape[0] : rhsShape[1];
  uint32_t acc = matmulOp.getAccumulate();
  fileOS << llvm::formatv(
      "{0,-14} desc={1:x} dst={2:x} lhs={3:x} rhs={4:x} "
      "M={5:d} N={6:d} K={7:d} acc={8:d} ta={9:d} tb={10:d}\n",
      "MATMUL", descAddr, dst, lhs, rhs, M, N, K, acc,
      static_cast<uint32_t>(transA), static_cast<uint32_t>(transB));

  return descAddr + DESC_LEN;
}

} // namespace mlir::eclipse
