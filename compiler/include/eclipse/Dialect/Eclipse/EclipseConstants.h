#ifndef ECLIPSE_DIALECT_ECLIPSE_CONSTANTS_H
#define ECLIPSE_DIALECT_ECLIPSE_CONSTANTS_H

#include <cstdint>

namespace mlir::eclipse {

constexpr uint32_t SRAM_MEMORY_SPACE = 0;
constexpr uint32_t DDR_MEMORY_SPACE = 1;
constexpr uint64_t SRAM_ALIGNMENT = 32;

} // namespace mlir::eclipse

#endif // ECLIPSE_DIALECT_ECLIPSE_CONSTANTS_H
