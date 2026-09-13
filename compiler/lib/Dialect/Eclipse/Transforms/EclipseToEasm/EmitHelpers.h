#ifndef ECLIPSE_TRANSFORMS_ECLIPSETOEASM_EMITHELPERS_H
#define ECLIPSE_TRANSFORMS_ECLIPSETOEASM_EMITHELPERS_H

#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir {
class Operation;

namespace eclipse {

/// 命令队列区里每个 descriptor 占的字节数。
constexpr uint32_t DESC_LEN = 0x40;

/// 各发射函数把一条指令（连同它的 descriptor）写进 .easm，并返回下一条
/// 指令可用的 descriptor 地址。按 op 家族拆在各 *Emit.cpp 里。
uint32_t emitDmaOp(Operation *op, llvm::raw_ostream &fileOS, uint32_t descAddr);
uint32_t emitSyncOp(Operation *op, llvm::raw_ostream &fileOS, uint32_t descAddr);
uint32_t emitMatmulOp(Operation *op, llvm::raw_ostream &fileOS,
                      uint32_t descAddr);
uint32_t emitEwiseOp(Operation *op, llvm::raw_ostream &fileOS, uint32_t descAddr);
uint32_t emitActOp(Operation *op, llvm::raw_ostream &fileOS, uint32_t descAddr);

} // namespace eclipse
} // namespace mlir

#endif // ECLIPSE_TRANSFORMS_ECLIPSETOEASM_EMITHELPERS_H
