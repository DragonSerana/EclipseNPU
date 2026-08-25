#include "SramAllocator.h"

namespace mlir::eclipse {

SramAllocator::SramAllocator(uint32_t base, uint32_t size)
    : base_(base), size_(size) {}

uint32_t SramAllocator::allocate(uint32_t size, uint32_t alignment) {

  return 0;
}

} // namespace mlir::eclipse
