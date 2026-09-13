#include "EmitHelpers.h"

#include "eclipse/Dialect/Eclipse/EclipseOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;

namespace mlir::eclipse {

namespace {

/// 四个 elementwise op 的 operand 约定一致（lhs, rhs, dst），只有 mnemonic 不同。
template <typename EwiseOpTy>
uint32_t emitEwise(StringRef mnemonic, EwiseOpTy ewiseOp,
                   llvm::raw_ostream &fileOS, uint32_t descAddr) {
  Value dstValue = ewiseOp.getDst();
  Value lhsValue = ewiseOp.getLhs();
  Value rhsValue = ewiseOp.getRhs();

  uint32_t dst = dstValue.getDefiningOp<SramOp>().getAddr();
  uint32_t lhs = lhsValue.getDefiningOp<SramOp>().getAddr();
  uint32_t rhs = rhsValue.getDefiningOp<SramOp>().getAddr();

  auto lhsType = mlir::cast<MemRefType>(lhsValue.getType());
  uint32_t n = 1;
  for (auto dim : lhsType.getShape())
    n *= dim;

  fileOS << llvm::formatv(
      "{0,-15} desc={1:x} dst={2:x} lhs={3:x} rhs={4:x} n={5:d}\n", mnemonic,
      descAddr, dst, lhs, rhs, n);

  return descAddr + DESC_LEN;
}

} // namespace

uint32_t emitEwiseOp(Operation *op, llvm::raw_ostream &fileOS,
                     uint32_t descAddr) {
  if (auto ewiseOp = mlir::dyn_cast<EwiseAddOp>(op))
    return emitEwise("ELEMENTWISE_ADD", ewiseOp, fileOS, descAddr);
  if (auto ewiseOp = mlir::dyn_cast<EwiseSubOp>(op))
    return emitEwise("ELEMENTWISE_SUB", ewiseOp, fileOS, descAddr);
  if (auto ewiseOp = mlir::dyn_cast<EwiseMulOp>(op))
    return emitEwise("ELEMENTWISE_MUL", ewiseOp, fileOS, descAddr);
  if (auto ewiseOp = mlir::dyn_cast<EwiseDivOp>(op))
    return emitEwise("ELEMENTWISE_DIV", ewiseOp, fileOS, descAddr);
  llvm_unreachable("emitEwiseOp: not an elementwise op");
}

} // namespace mlir::eclipse
